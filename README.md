# Manga Mood Music

A pipeline that analyzes manga pages and matches background music to the emotional tone of the scene — because generic playlists never quite match a chill scene, a tense fight, or a gut-punch moment.

## How it works (current state)

1. **Local proxy server** (`proxy_server.py`) — manga page images are hosted on a CDN that blocks direct browser access via CORS. A small Flask server fetches image bytes server-side (where CORS doesn't apply) and relays them back.
2. **CLIP-based mood classifier** (`clip_mood_classifier.py`) — uses OpenAI's CLIP model (zero-shot, via Hugging Face `transformers`) to score a manga page against candidate mood labels (e.g. "tense buildup scene," "emotional sad scene") based on visual similarity.

## What's next

- **OCR + text sentiment** — extract dialogue text from panels and score it separately for mood, since CLIP doesn't meaningfully read in-image text and misses a lot of emotional signal that's only present in dialogue.
- **Weighted signal combination** — combine the image-based and text-based mood scores into a single prediction, with weights tuned against a small hand-labeled test set.
- **Music selection layer** — map final mood to actual playback (e.g. Spotify search/queue).

## Known limitations

CLIP was trained primarily on real-world photographs, not stylized manga art. In testing, it showed a strong bias toward classifying visually intense panels (wide mouths, heavy linework, exaggerated expressions) as "sad," even when the actual content was comedic — likely because that visual grammar more often signals distress in CLIP's training data. This is part of the motivation for adding a text-based signal alongside the visual one.

## Setup

Requires Python 3.12 (PyTorch dropped Intel Mac support after v2.2.2 — if you're on an Intel Mac, you'll need this specific version too).

\`\`\`bash
python3.12 -m venv venv_manga
source venv_manga/bin/activate
pip install torch==2.2.2 transformers==4.40.0 "numpy<2" pillow requests flask
\`\`\`

Run the proxy server in one terminal:
\`\`\`bash
python proxy_server.py
\`\`\`

Run the classifier in another (proxy must be running first):
\`\`\`bash
python clip_mood_classifier.py
\`\`\`

## Note on manga sourcing

This project currently reads pages from a scanlation aggregator for local prototyping/testing purposes only. Not intended for redistribution or public deployment against that source.
