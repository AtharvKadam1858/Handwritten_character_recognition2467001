# utils/generator.py
import os
import random
from PIL import Image, ImageDraw, ImageFont

def generate_dataset(output_folder='dataset/images', label_file='dataset/labels.txt', count=500):
    os.makedirs(output_folder, exist_ok=True)
    words = ["hello", "world", "openai", "machine", "learning", "handwriting", "recognition", "streamlit", "python", "model", "deep", "neural", "network"]
    font = ImageFont.truetype("arial.ttf", 28)  # Change to a handwritten font if available

    with open(label_file, 'w') as f:
        for i in range(count):
            text = " ".join(random.choices(words, k=random.randint(1, 4)))
            img = Image.new('L', (200, 50), color=255)
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), text, font=font, fill=0)

            path = os.path.join(output_folder, f"img_{i}.png")
            img.save(path)
            f.write(f"{path}\t{text}\n")

    print(f"✅ Generated {count} images at {output_folder}")
