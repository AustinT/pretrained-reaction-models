import pickle

with open("./routes_possible_test_hard.pkl", "rb") as f:
    data = pickle.load(f)

test_smiles = [item[0].split(">>")[0] for item in data]
with open("./retro-star-test-mols.smiles", "w") as f:
    f.write("\n".join(test_smiles))
