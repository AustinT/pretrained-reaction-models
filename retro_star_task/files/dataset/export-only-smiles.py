"""Script to export retro* canonical SMILES to .smiles file"""

from tqdm import tqdm

with open("./origin_dict-canonical.csv") as f:
    with open("retro-star-inventory.smiles", "w") as f_out:
        next(f)  # discard header
        for line in tqdm(f):
            tokens = line.strip().split(",")
            f_out.write(tokens[1] + "\n")
