# Script to canonicalize the origin dict
from pathlib import Path
from tqdm.auto import tqdm
from rdkit import Chem
from joblib import Parallel, delayed

input_file = Path("./origin_dict.csv")
output_file = Path("./origin_dict-canonical.csv")

if __name__ == "__main__":
    print("Start of script.")

    print("Reading origin dict")
    with open(input_file) as f:
        input_file_contents = f.readlines()
    rows = [
        line.split(",")
        for line in tqdm(input_file_contents[1:], desc="Splitting lines")
    ]
    old_smiles = [r[1] for r in rows]

    print("Doing canonicalization.")
    new_smiles = Parallel(n_jobs=6)(
        delayed(Chem.CanonSmiles)(s) for s in tqdm(old_smiles, desc="Canonicalizing.")
    )
    print(
        "Canonicalization done."
        f" {len([ns for ns, os in zip(new_smiles, old_smiles) if ns != os])}/{len(new_smiles)} changed."
    )

    print("Writing output file")
    output_lines = input_file_contents[:1]
    for i, r in enumerate(rows):
        r[1] = new_smiles[i]
        output_lines.append(",".join(r))
    with open(output_file, "w") as f:
        for line in output_lines:
            f.write(line)
    print("END OF SCRIPT.")
