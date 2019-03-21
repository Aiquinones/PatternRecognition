from reconocedorSC import reconocedorSC, get_matrix_from_txt
from constants import letters
from os import listdir
from os.path import isfile, join

mypath = "TestSet/pngs/C"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

score = 0
tested = 0

for letter in letters:
    mypath = f"TestSet/binaries/{letter}"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in onlyfiles:
        tested += 1

        M = get_matrix_from_txt(f"{mypath}/{file}")
        pred = reconocedorSC(M)

        if (letter == "C" and pred == 0) or (letter == "S" and pred == 1):
            score += 1
            print("match!")
        else:
            print("miss..")

print(f"{score*100/tested}%")
