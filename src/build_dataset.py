import json, shutil, sys
from collections import Counter
from pathlib import Path
# Builds <dataset>/<class>/*.jpg from labels.jsonl. Makes no API calls, so edit these settings and rerun for free.
TARGET_QUESTION = "mood"  # any question name from label_pages.py, e.g. "music_genre", "energy", "character_emotion"
# answer option -> output class; options left out are dropped and their probability is ignored
CLASS_GROUPS = {
    "happy": "happy", "funny": "happy", "romantic": "happy",
    "calm": "calm", "neutral": "calm",
    "sad": "sad",
    "tense": "tense", "action": "tense", "angry": "tense",
    "scary": "scary", "mysterious": "scary",
}
MINIMUM_CLASS_PROBABILITY = 0.5
# pilot: two-page spreads scored ~0.95 mixed on every page, so this filter is off by default
MAXIMUM_MIXED_MOOD_PROBABILITY = 1.0
PAGES_PER_CLASS = 100

labels_file, pages_folder, dataset_folder = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
# must be a new folder, so pages from an older grouping never mix into this one
dataset_folder.mkdir(parents=True)
pages_per_class = Counter()
for line in labels_file.open():
    record = json.loads(line)
    answers = {answer["name"]: answer for answer in record["decision"]["answers"]}
    target = answers[TARGET_QUESTION]
    if target["type"] == "refusal" or answers["mixed_mood"].get("probability", 0) > MAXIMUM_MIXED_MOOD_PROBABILITY:
        continue
    class_probability = Counter()
    for option in target["probabilities"]:
        # score options carry a "label", choice options only a "value"
        option_name = option.get("label") or option["value"]
        if option_name in CLASS_GROUPS:
            class_probability[CLASS_GROUPS[option_name]] += option["probability"]
    if not class_probability:
        continue
    page_class, probability = class_probability.most_common(1)[0]
    if probability < MINIMUM_CLASS_PROBABILITY or pages_per_class[page_class] >= PAGES_PER_CLASS:
        continue
    (dataset_folder / page_class).mkdir(exist_ok=True)
    shutil.copy(pages_folder / record["page"], dataset_folder / page_class)
    pages_per_class[page_class] += 1
print(dict(pages_per_class))
