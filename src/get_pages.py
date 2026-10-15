import random, shutil, sys
from pathlib import Path
# Manga109-s (hal-utokyo/Manga109-s on Hugging Face, gated): images/<book>/<page>.jpg, each image is a two-page spread
manga109_folder, pages_wanted, output_folder = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
output_folder.mkdir(exist_ok=True)
all_pages = list((manga109_folder / "images").glob("*/*.jpg"))
# random sample across all 87 books so no single title dominates a mood
for page_path in random.sample(all_pages, min(pages_wanted, len(all_pages))):
    shutil.copy(page_path, output_folder / f"{page_path.parent.name}_{page_path.name}")
