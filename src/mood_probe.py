import sys, joblib, open_clip, torch
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# 512px SigLIP2 over 224px CLIP: speech-bubble lettering stays legible without OCR
image_model, _, preprocess_image = open_clip.create_model_and_transforms("ViT-B-16-SigLIP2-512", pretrained="webli")

@torch.no_grad()
def images_to_vectors(image_paths):
    vectors = [image_model.encode_image(preprocess_image(Image.open(path).convert("RGB")).unsqueeze(0), normalize=True)
               for path in image_paths]
    return torch.cat(vectors).numpy()

command, target_path = sys.argv[1], sys.argv[2]
if command == "train":
    image_paths = list(Path(target_path).glob("*/*"))
    image_vectors = images_to_vectors(image_paths)
    mood_labels = [path.parent.name for path in image_paths]
    accuracy = cross_val_score(LogisticRegression(max_iter=2000), image_vectors, mood_labels).mean()
    print("cross-validated accuracy:", round(accuracy, 3))
    joblib.dump(LogisticRegression(max_iter=2000).fit(image_vectors, mood_labels), "mood_classifier.joblib")
else:
    mood_classifier = joblib.load("mood_classifier.joblib")
    mood_probabilities = mood_classifier.predict_proba(images_to_vectors([target_path]))[0]
    for mood, probability in zip(mood_classifier.classes_, mood_probabilities):
        print(f"{mood}: {probability:.3f}")
