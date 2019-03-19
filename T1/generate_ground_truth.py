from constants import fonts, letters
from preprocess import get_preprocessed_saveable
import cv2

for letter in letters:
    for i, font in enumerate(fonts):
        filepath = f"Fonts/{font}/{letter}.png"
        saveable = get_preprocessed_saveable(filepath)
        cv2.imwrite(f"GroundTruth/{letter}{i}.png", saveable)
