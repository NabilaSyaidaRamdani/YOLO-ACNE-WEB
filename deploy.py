from ultralytics import YOLO
from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 1. Load model
model = YOLO("best.pt")

# 2. Predict image
results = model("WHITEHEADS DAN PUSTULE.jpg")[0]

# 3. Count classes
class_ids = results.boxes.cls.cpu().numpy()
class_names = results.names
counts = Counter(class_ids)

# 4. Get the image with bounding boxes
img_with_boxes = results.plot()

# 5. Convert to PIL Image
img_pil = Image.fromarray(img_with_boxes)

# 6. Add class counts using PIL
draw = ImageDraw.Draw(img_pil)

# Optional: set font (in Streamlit might need fallback font)
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

y_offset = 30
for i, (class_id, count) in enumerate(counts.items()):
    name = class_names[int(class_id)]
    label = f"{name}: {count}"
    draw.text((10, y_offset + i * 30), label, font=font, fill=(255, 0, 0))

# 7. Save and show
img_pil.save("output_dengan_nama_dan_count.jpg")

# Show with matplotlib
plt.imshow(img_pil)
plt.axis('off')
plt.show()
