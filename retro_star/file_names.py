"""Store data about file names."""

from pathlib import Path

BASE_PATH = Path(__file__).parent.absolute() / "files"

INVENTORY_CSV = str(BASE_PATH / "dataset" / "origin_dict.csv")
MODEL_CHECKPOINT = str(BASE_PATH / "one_step_model" / "saved_rollout_state_1_2048.ckpt")
TEMPLATES = str(BASE_PATH / "one_step_model" / "template_rules_1.dat")
TEST_ROUTES = str(BASE_PATH / "dataset" / "routes_possible_test_hard.pkl")
ALL = [INVENTORY_CSV, MODEL_CHECKPOINT, TEMPLATES, TEST_ROUTES]

all_present = all(Path(p).exists() for p in ALL)
