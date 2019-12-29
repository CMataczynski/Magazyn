
# Pobieranie węzłów jako dict

def load_img2graphdict(filename):
    i2gdict = {}
    with open(filename, mode="r") as file:
        for line in file:
            idx, rest = line.strip().strip(";").split(":")
            idx = int(idx)
            x, y = rest.split(",")
            x = int(x)
            y = int(y)
            i2gdict[idx] = (x, y)
    return i2gdict