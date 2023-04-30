"""Handy script to check that SMILES in origin dict are canonical via random sampling."""

import argparse
import random

from rdkit import Chem

if __name__ == "__main__":

    print("Start script.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csv_file",
    )
    parser.add_argument("--sample_prob", type=float, default=1e-4)
    args = parser.parse_args()

    # Randomly sample SMILES
    smiles_list = []
    with open(args.input_csv_file) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            elif random.random() < args.sample_prob:
                smiles_list.append(line.strip().split(",")[1])
    print(f"Randomly sampled {len(smiles_list)} SMILES.")

    # Canonicalize SMILES
    num_non_canon = 0
    for s in smiles_list:
        s_canon = Chem.CanonSmiles(s)
        if s != s_canon:
            num_non_canon += 1
            print(f"Found {s} =/= canonical {s_canon}")

    print(
        f"End of script. Found {num_non_canon}/{len(smiles_list)} non-canonical SMILES."
    )
