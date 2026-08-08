import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPModel, CLIPProcessor

# load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# define mood labels
mood_labels = ["fighting", "crying", "ominous entrance", "laughing", "shouting in anger", "sitting alone", "looking longinly"]

# fetch manga page image
proxy_url = "http://127.0.0.1:5001/fetch-image?url=https://hot.planeptune.us/manga/Naruto/0001-035.png"
response = requests.get(proxy_url)

# turn raw bites into a PIL image
image = Image.open(BytesIO(response.content))

inputs = processor(text=mood_labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)

# to see results
for label, prob in zip(mood_labels, probs[0].tolist()):
    print(f"{label}: {prob:.2%}")
