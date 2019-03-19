import numpy as np
import cv2
from constants import height, white_limit, width


def get_binary(img):
    bin = np.ndarray((height, width))
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            bin[i][j] = 0 if np.mean(pixel) >= white_limit else 1
    return bin


def get_saveable(bin):
    assert (len(bin) > 0), "Can't transform an empty binary picture"
    img = np.ndarray((len(bin), len(bin[0]), 3))
    for i, row in enumerate(bin):
        for j, pixel in enumerate(row):
            img[i][j] = np.array([255, 255, 255]) if pixel == 0 else np.array(
                [0, 0, 0])
    return img


def easy_print(bin):
    for row in bin:
        st = ""
        for pixel in row:
            c = "-" if pixel == 0 else "*"
            st += c
        print(st)


def save_bin(bin, filepath):
    ans = ""
    for row in bin:
        st = ""
        for pixel in row:
            c = "0" if pixel == 0 else "1"
            st += c
        ans += f"{st}\n"
    with open(filepath, "w") as file:
        file.write(ans)


def crop(bin):
    h = len(bin)
    w = len(bin[0])

    top = 0
    bottom = h - 1
    left = 0
    right = w - 1

    # fill top
    top_seen = False
    for i, row in enumerate(bin):
        for j, pixel in enumerate(row):
            if pixel == 1:
                top = i
                top_seen = True
                break
        if top_seen:
            break

    # fill left
    left_seen = False  # reduces execution time
    for i, row in enumerate(bin):
        for j, pixel in enumerate(row):
            if left_seen and j >= left:
                break
            if pixel == 1:
                left = j
                left_seen = True

    # fill right
    right_seen = False
    for i, row in enumerate(bin):
        left_border = right if right_seen else 0  # reduces execution time
        j = w - 1
        while left_border < j:
            pixel = row[j]
            if pixel == 1:
                right_seen = True
                right = j
                break
            j -= 1

    # fill bottom
    i = h - 1
    bottom_seen = False
    while i >= 0:
        row = bin[i]
        for pixel in row:
            if pixel == 1:
                bottom = i
                bottom_seen = True
                break
        if bottom_seen:
            break
        i -= 1

    return bin[top:bottom, left:right]


def get_preprocessed_saveable(filename, easy_pr=False):
    img = cv2.imread(filename)
    res = cv2.resize(img, dsize=(height, width))
    binary = get_binary(res)
    cropped = crop(binary)
    if easy_pr:
        easy_print(cropped)
        save_bin(cropped, "TestSet/test2.txt")
    return get_saveable(cropped)


def preprocess(X):
    bin = np.array(X)
    cropped = crop(bin)
    img = get_saveable(cropped)
    res = cv2.resize(img, dsize=(height, width))
    return get_binary(res)


if __name__ == '__main__':
    filename = 'Fonts/Arial/C.png'
    ans = get_preprocessed_saveable(filename)

    cv2.imshow('test', ans)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.destroyAllWindows()
        cv2.imwrite("c.png", ans)
