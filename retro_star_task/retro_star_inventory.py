from __future__ import annotations
import csv

from syntheseus.search.chem import Molecule
from syntheseus.search.mol_inventory import BaseMolInventory

from . import file_names


class RetroStarInventory(BaseMolInventory):
    """A custom inventory object because the inventory is quite large."""

    def __init__(
        self,
        inventory_csv: str = file_names.INVENTORY_CSV,
        canonicalize: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        with open(inventory_csv, "r") as f:
            reader = csv.reader(f)
            next(reader)  # discard header
            smiles_list = [row[1].strip() for row in reader]

        if canonicalize:
            raise NotImplementedError

        self._smiles_set = set(smiles_list)

    def is_purchasable(self, mol: Molecule) -> bool:
        return mol.smiles in self._smiles_set
