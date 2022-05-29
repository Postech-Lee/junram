import os
import glob
from PIL import Image

files = glob.glob('C://Users/user/PycharmProjects/datasets/fake_picture/*')

for f in files:
    title, ext = os.path.splitext(f)
    if ext in ['.jpg', '.png']:
        img = Image.open(f)
        img_resize = img.resize((256,256))
        img_resize.save(title + ext)
        print(title)