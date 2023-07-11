from __future__ import annotations

import pickle

from retro_star_task import file_names


def get_190_hard_test_smiles() -> list[str]:
    with open(file_names.TEST_ROUTES, "rb") as f:
        test_routes = pickle.load(f)
    output = [r[0].split(">")[0] for r in test_routes]
    assert len(output) == 190
    return output
