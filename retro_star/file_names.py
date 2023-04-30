"""Store data about file names."""

from pathlib import Path

BASE_PATH = Path(__file__).parent.absolute() / "files"

INVENTORY_CSV = str(BASE_PATH / "dataset" / "origin_dict-canonical.csv")
RXN_MODEL_CHECKPOINT = str(
    BASE_PATH / "one_step_model" / "saved_rollout_state_1_2048.ckpt"
)
VALUE_MODEL_CHECKPOINT = str(BASE_PATH / "saved_models" / "best_epoch_final_4.pt")
TEMPLATES = str(BASE_PATH / "one_step_model" / "template_rules_1.dat")
TEST_ROUTES = str(BASE_PATH / "dataset" / "routes_possible_test_hard.pkl")
ALL = [
    INVENTORY_CSV,
    RXN_MODEL_CHECKPOINT,
    TEMPLATES,
    TEST_ROUTES,
    VALUE_MODEL_CHECKPOINT,
]

all_present = all(Path(p).exists() for p in ALL)
