from preprocess import get_binary, save_bin
from constants import letters
import cv2

from os import listdir
from os.path import isfile, join

mypath = "TestSet/pngs/C"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for letter in letters:
    mypath = f"TestSet/pngs/{letter}"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in onlyfiles:
        filename = file[:-4]

        img = cv2.imread(f"{mypath}/{file}")
        bin = get_binary(img, limited=False)
        save_bin(bin, f"TestSet/binaries/{letter}/{filename}.txt")
