import PIL
from PIL import Image
import os

img_width = 300
img_height = 300

dirList = {"fruit", "vegetable", "egg", "meat", "condiments", "dairy"}

for d in dirList:
	dirname = os.fsencode(d)
	for f in os.listdir(d):
		filename = os.fsdecode(f)
		if filename.endswith(".jpg"):
			im1 = Image.open(d + "/" + f)
			im2 = im1.resize((img_width, img_height), Image.NEAREST)
			im3 = im2.convert("RGB")
			im3.save(d + "/" + f)
			continue
		else: 
			continue
