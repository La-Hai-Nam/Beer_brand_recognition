from PIL import Image
import os

path = "/home/leon/Documents/biererkennung/beer_ds/beer/warsteiner/"
dirs = os.listdir(path)

def resize():
    for item in dirs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            f, e = os.path.splitext(path + item)
            imResize = im.resize((150, 150), Image.ANTIALIAS)
            imResize.save(f+".jpg", 'JPEG', quality=90)
            print(f)


resize()
