import base64, json, sys
from pathlib import Path
from openai import OpenAI

def choice(name, instructions, options):
    return {"type": "choice", "name": name, "instructions": instructions,
            "choices": [{"value": value, "description": description} for value, description in options.items()]}

def score(name, instructions, levels):
    return {"type": "score", "name": name, "instructions": instructions, "levels": [{"label": label} for label in levels]}

# Paid once, reused forever: build_dataset.py regroups, filters and balances these answers offline.
# Fine-grained options on purpose, because merging later is free but splitting later needs new API calls.
# The image is the expensive input and is sent once per page, so extra questions add only a little text.
QUESTIONS = [
    choice("mood", "Overall mood of this manga page, judging from the art and the dialogue?", {
        "happy": "cheerful, warm, celebratory",
        "funny": "comedy, gag, playful",
        "calm": "quiet, everyday, peaceful",
        "romantic": "love, affection, longing",
        "sad": "grief, loneliness, regret",
        "tense": "suspense, pressure, uneasy",
        "action": "fight, chase, fast movement",
        "angry": "rage, shouting, hostility",
        "scary": "horror, dread, threat",
        "mysterious": "eerie, unexplained, ominous",
        "neutral": "exposition, no clear mood",
    }),
    # Ekman's 6 emotions plus neutral, the same classes as the KangaiSet manga face dataset, so results compare with papers
    choice("character_emotion", "Main emotion shown on the characters' faces on this page?", {
        "anger": "", "disgust": "", "fear": "", "happiness": "", "sadness": "", "surprise": "", "neutral": "",
        "no_faces": "no character faces visible",
    }),
    # energy and valence are the two axes music is commonly described by, so they map straight onto music search
    score("energy", "How intense or energetic is this page?", ["very calm", "calm", "medium", "intense", "very intense"]),
    score("valence", "How positive or negative does this page feel?", ["very negative", "negative", "neutral", "positive", "very positive"]),
    choice("music_genre", "Which background music style would best fit this page?", {
        "upbeat_pop": "", "acoustic": "", "lofi": "", "piano": "", "orchestral_epic": "", "rock": "",
        "electronic": "", "jazz": "", "ambient": "", "horror_soundtrack": "", "comedy_soundtrack": "",
    }),
    score("tempo", "How fast should the background music be?", ["very slow", "slow", "medium", "fast", "very fast"]),
    choice("scene_type", "What kind of scene is this page mostly?", {
        "dialogue": "characters talking", "action": "physical movement or fighting",
        "inner_thoughts": "a character thinking or remembering", "establishing": "setting or scenery",
        "comedy_gag": "a joke or slapstick moment",
    }),
    {"type": "predicate", "name": "mixed_mood", "instructions": "Do different panels on this page show clearly different moods?"},
    {"type": "predicate", "name": "readable_text_matters", "instructions": "Is the dialogue needed to understand the mood of this page?"},
]

pages_folder, labels_file = Path(sys.argv[1]), Path(sys.argv[2])
# one JSON line per page, written as soon as it is paid for, so a crashed or stopped run resumes without paying twice
already_labelled = {json.loads(line)["page"] for line in labels_file.open()} if labels_file.exists() else set()
openai_client = OpenAI()
with labels_file.open("a") as labels:
    for page_path in sorted(pages_folder.glob("*.jpg")):
        if page_path.name in already_labelled:
            continue
        page_as_data_url = "data:image/jpeg;base64," + base64.b64encode(page_path.read_bytes()).decode()
        # Decisions API (public beta, Oct 2026): gpt-6-luna is the only supported model
        decision = openai_client.decisions.create(
            model="gpt-6-luna",
            input=[{"role": "user", "content": [{"type": "input_image", "image_url": page_as_data_url}]}],
            questions=QUESTIONS,
        )
        labels.write(json.dumps({"page": page_path.name, "decision": decision.model_dump()}) + "\n")
        labels.flush()
