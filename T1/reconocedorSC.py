import sys
from constants import width, height, fonts, letters
from preprocess import preprocess
import numpy as np


# Obten la matriz a partir del filepath .npy
def get_matrix(filepath):
    return np.load(filepath)


# Obten la matriz a partir del filepath .txt
def get_matrix_from_txt(filepath):
    matrix = []
    with open(filepath, 'r') as file:
        for line in [l.strip() for l in file.readlines()]:
            row = [int(pixel) for pixel in line]
            matrix.append(row)
    return matrix


# itera en los filtros y considera como correcto la letra del filtro con mayor
# precision (score).
def reconocedorSC(mat, printing=False):
    mat = preprocess(mat)

    best_score = 0
    best = None

    for letter in letters:
        for i, font in enumerate(fonts):
            filepath = f"GroundTruth/binaries/{letter}{i}.txt"
            img = get_matrix_from_txt(filepath)

            score = 0
            for i in range(height):
                for j in range(width):
                    add = 1 if img[i][j] == mat[i][j] else 0
                    score += add
            score /= height * width
            score *= 100

            if printing:
                print(f"{letter} {font}: {score}%")

            if score > best_score:
                best_score = score
                best = letter

    assert best, 'No se reconoce preferencia entre S y C'
    return 1 if best == 'S' else 0


if __name__ == '__main__':
    assert len(sys.argv) > 1, "filepath needed"

    mat = get_matrix(sys.argv[1])
    print(reconocedorSC(mat))
