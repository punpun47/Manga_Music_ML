"""Soft-label panel-pooled mood probe. Commands: extract | train | eval | baseline."""
import argparse
import json
import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as probability_function
from PIL import Image
from sklearn.metrics import classification_report
from torch import nn

TARGET_MOODS = ["happy", "calm", "sad", "tense"]
# 11 API moods -> 4 targets; merging is free, splitting later needs new API calls
FINE_TO_TARGET = {
    "happy": "happy", "funny": "happy", "romantic": "happy",
    "calm": "calm", "neutral": "calm", "sad": "sad",
    "tense": "tense", "action": "tense", "angry": "tense", "scary": "tense", "mysterious": "tense",
}
MOOD_KEYWORDS = {
    "happy": ["happy pop", "upbeat", "feel-good"],
    "calm": ["lofi", "calm acoustic", "chill"],
    "sad": ["sad acoustic", "piano ballad", "melancholy"],
    "tense": ["epic action", "intense", "anime battle"],
}
FILENAME_PATTERN = re.compile(r"^(.+)_(\d+)\.jpg$")


def load_soft_labels(labels_path):
    soft_labels = {}
    for line in Path(labels_path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        answers = {answer["name"]: answer for answer in record["decision"]["answers"]}
        grouped = {mood: 0.0 for mood in TARGET_MOODS}
        for option in answers["mood"]["probabilities"]:
            grouped[FINE_TO_TARGET[option["value"]]] += option["probability"]
        total = sum(grouped.values())
        soft_labels[record["page"]] = {mood: value / total for mood, value in grouped.items()}
    return soft_labels


def load_panel_boxes(manga109_root, book, page_index):
    root = ElementTree.parse(Path(manga109_root) / "annotations" / f"{book}.xml").getroot()
    for page in root.iter("page"):
        if int(page.get("index")) == page_index:
            return [tuple(int(frame.get(side)) for side in ("xmin", "ymin", "xmax", "ymax"))
                    for frame in page.iter("frame")]
    return []


def embed_panels(model, preprocess, spread_path, panel_boxes, device):
    spread_image = Image.open(spread_path).convert("RGB")
    crops = [spread_image.crop(box) for box in panel_boxes] or [spread_image]
    vectors = []
    with torch.no_grad():
        for start in range(0, len(crops), 16):
            batch = torch.stack([preprocess(crop) for crop in crops[start:start + 16]]).to(device)
            features = model.encode_image(batch)
            vectors.append((features / features.norm(dim=-1, keepdim=True)).float().cpu())
    return torch.cat(vectors).numpy()


def pool_panels(panel_vectors, method):
    stacked = torch.as_tensor(np.asarray(panel_vectors, dtype=np.float32))
    mean_pooled = stacked.mean(dim=0)
    if method == "mean":
        return mean_pooled.numpy()
    max_pooled = stacked.max(dim=0).values
    if method == "max":
        return max_pooled.numpy()
    return torch.cat([mean_pooled, max_pooled]).numpy()  # concat


def extract_all_spreads(pooling, labels_path, pages_folder, manga109_root, cache_path, device):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-SigLIP2-256", pretrained="webli", device=device)
    model.eval()
    soft_labels = load_soft_labels(labels_path)
    records, vectors = [], []
    for filename in sorted(soft_labels):
        book, page_index = FILENAME_PATTERN.match(filename).groups()
        panel_boxes = load_panel_boxes(manga109_root, book, int(page_index))
        vectors.append(pool_panels(embed_panels(model, preprocess, Path(pages_folder) / filename, panel_boxes, device), pooling))
        records.append({"filename": filename, "book": book})
    torch.save({"vectors": np.stack(vectors).astype(np.float32), "records": records}, cache_path)
    print(f"cached {len(records)} spreads -> {cache_path}")


class MoodProbe(nn.Module):
    def __init__(self, input_size, use_mlp):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(input_size, 256), nn.LayerNorm(256), nn.GELU(),
                                  nn.Dropout(0.4), nn.Linear(256, 4)) if use_mlp else nn.Linear(input_size, 4)

    def forward(self, spread_vectors):
        return self.head(spread_vectors)


def title_grouped_split(records, seed=42):
    books = sorted({record["book"] for record in records})
    np.random.RandomState(seed).shuffle(books)
    test_books, test_count = set(), 0
    for book in books:
        if test_count >= len(records) * 0.2:
            break
        test_books.add(book)
        test_count += sum(1 for record in records if record["book"] == book)
    return ([i for i, r in enumerate(records) if r["book"] not in test_books],
            [i for i, r in enumerate(records) if r["book"] in test_books])


def fit_probe(vectors, targets, input_size, use_mlp, loss_name, device, seed=42, class_weights=None, temperature=1.0):
    torch.manual_seed(seed)
    probe = MoodProbe(input_size, use_mlp).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3, weight_decay=1e-2)
    train_vectors, train_targets = vectors.to(device), targets.to(device)
    holdout = torch.randperm(len(vectors))[:max(1, len(vectors) // 10)]
    mask = torch.ones(len(vectors), dtype=torch.bool)
    mask[holdout] = False
    best_loss, stalled = float("inf"), 0
    for _ in range(200):
        probe.train()
        for start in range(0, mask.sum().item(), 64):
            batch = torch.where(mask)[0][start:start + 64]
            batch_loss = soft_loss(probe(train_vectors[batch]), train_targets[batch], loss_name, class_weights, temperature)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        probe.eval()
        with torch.no_grad():
            validation_loss = soft_loss(probe(train_vectors[holdout]), train_targets[holdout], loss_name, class_weights, temperature).item()
        if validation_loss < best_loss - 1e-4:
            best_loss, stalled = validation_loss, 0
            best_state = {k: v.cpu() for k, v in probe.state_dict().items()}
        else:
            stalled += 1
            if stalled >= 15:
                break
    probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return probe


def soft_loss(predicted_logits, target_distributions, loss_name, class_weights=None, temperature=1.0):
    # distillation temperature: soften both sides so secondary moods teach
    log_probabilities = probability_function.log_softmax(predicted_logits / temperature, dim=-1)
    if temperature != 1.0:
        target_distributions = target_distributions ** (1.0 / temperature)
        target_distributions = target_distributions / target_distributions.sum(dim=-1, keepdim=True)
    if loss_name == "kl":
        if class_weights is None:
            base_loss = nn.KLDivLoss(reduction="batchmean")(log_probabilities, target_distributions)
        else:
            # minority classes speak louder: reweight each class's contribution
            weights = torch.as_tensor(class_weights, dtype=log_probabilities.dtype, device=log_probabilities.device)
            base_loss = -(weights * target_distributions * log_probabilities).sum(dim=-1).mean()
        return base_loss * temperature * temperature  # Hinton scaling keeps gradients comparable across T
    return nn.CrossEntropyLoss()(predicted_logits, target_distributions.argmax(dim=-1))


def score_predictions(predicted, targets):
    true_hard, pred_hard = targets.argmax(axis=1), predicted.argmax(axis=1)
    report = classification_report(true_hard, pred_hard, target_names=TARGET_MOODS, output_dict=True, zero_division=0)
    return {
        "accuracy": float((pred_hard == true_hard).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class_f1": {mood: float(report[mood]["f1-score"]) for mood in TARGET_MOODS},
        "mean_absolute_error": float(np.abs(predicted - targets).mean()),
        "mean_kl": float((targets * np.log((targets + 1e-9) / (predicted + 1e-9))).sum(axis=1).mean()),
    }


def load_split_data(cache_path, labels_path):
    cache = torch.load(cache_path, weights_only=False)
    soft_labels = load_soft_labels(labels_path)
    order = [record["filename"] for record in cache["records"]]
    vectors = torch.as_tensor(cache["vectors"])
    targets = torch.as_tensor([[soft_labels[f][m] for m in TARGET_MOODS] for f in order])
    train_indices, test_indices = title_grouped_split(cache["records"])
    return vectors, targets, order, train_indices, test_indices


def train_command(cache_path, labels_path, pooling, use_mlp, loss_name, output_folder, device, use_weights=False, temperature=1.0):
    vectors, targets, order, train_indices, test_indices = load_split_data(cache_path, labels_path)
    class_weights = None
    if use_weights:
        # inverse frequency of grouped hard labels: rare moods count more
        hard_labels = targets[train_indices].argmax(dim=-1).numpy()
        counts = np.bincount(hard_labels, minlength=len(TARGET_MOODS)).astype(float)
        class_weights = (len(hard_labels) / (len(TARGET_MOODS) * np.maximum(counts, 1))).tolist()
        print("class weights:", dict(zip(TARGET_MOODS, [round(w, 2) for w in class_weights])))
    probe = fit_probe(vectors[train_indices], targets[train_indices], vectors.shape[1], use_mlp, loss_name, device, class_weights=class_weights, temperature=temperature)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), output_folder / "best.pt")
    report = evaluate_checkpoint(vectors, targets, order, test_indices, probe, device)
    (output_folder / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


@torch.no_grad()
def evaluate_checkpoint(vectors, targets, order, test_indices, probe, device):
    probe.eval()
    predicted = probability_function.softmax(probe(vectors[test_indices].to(device)), dim=-1).cpu().numpy()
    return {"test_n": len(test_indices), **score_predictions(predicted, targets[test_indices].numpy())}


def format_spotify_query(predicted_distribution):
    (top_mood, top_probability), (second_mood, second_probability) = sorted(predicted_distribution.items(), key=lambda item: -item[1])[:2]
    total = top_probability + second_probability
    keywords = MOOD_KEYWORDS[top_mood][:max(1, round(3 * top_probability / total))] + MOOD_KEYWORDS[second_mood][:max(1, round(2 * second_probability / total))]
    return " ".join(dict.fromkeys(keywords))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "train", "eval", "baseline"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--labels", default="labels.jsonl")
        sub.add_argument("--pages", default="pages")
        sub.add_argument("--manga109", default="manga109/Manga109s_released_2026_05_21")
        sub.add_argument("--cache", default="soft_panel_features.pt")
        sub.add_argument("--pool", default="concat", choices=["mean", "max", "concat"])
        sub.add_argument("--probe", default="mlp", choices=["linear", "mlp"])
        sub.add_argument("--loss", default="kl", choices=["kl", "hard"])
        sub.add_argument("--weighted", action="store_true")
        sub.add_argument("--temperature", type=float, default=1.0)
        sub.add_argument("--out", default="runs/soft1")
        sub.add_argument("--checkpoint", default="runs/soft1/best.pt")
        sub.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    use_mlp = args.probe == "mlp"
    if args.command == "extract":
        extract_all_spreads(args.pool, args.labels, args.pages, args.manga109, args.cache, args.device)
    elif args.command in ("train", "baseline"):
        loss_name = "hard" if args.command == "baseline" else args.loss
        train_command(args.cache, args.labels, args.pool, use_mlp, loss_name, args.out, args.device, args.weighted, args.temperature)
    elif args.command == "eval":
        vectors, targets, order, _, test_indices = load_split_data(args.cache, args.labels)
        device = torch.device(args.device)
        probe = MoodProbe(vectors.shape[1], use_mlp).to(device)
        probe.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(json.dumps(evaluate_checkpoint(vectors, targets, order, test_indices, probe, device), indent=2))


if __name__ == "__main__":
    main()
