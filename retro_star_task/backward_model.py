from __future__ import annotations
from typing import Optional

import numpy as np
from rdkit import RDLogger
from syntheseus.search.chem import Molecule, BackwardReaction
from syntheseus.search.reaction_models import BackwardReactionModel
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator
import torch

from .retro_star_code.mlp_inference import MLPModel
from . import file_names

DEFAULT_RETROSTAR_EXPANSION_TOPk = 50

# Turn off rdkit logger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class RetroStarReactionModel(BackwardReactionModel):
    def __init__(
        self,
        model_checkpoint: str = file_names.RXN_MODEL_CHECKPOINT,
        template_file: str = file_names.TEMPLATES,
        expansion_topk: int = 50,
        device: Optional[int] = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.expansion_topk = expansion_topk
        if device is None:
            # Smart default: CUDA if it is available
            device = 0 if torch.cuda.is_available() else -1
        self.model = MLPModel(model_checkpoint, template_file, device=device)

    def _get_backward_reactions(
        self, mols: list[Molecule]
    ) -> list[list[BackwardReaction]]:
        output = []
        for mol in mols:
            curr_output = []

            # Call model
            output_dict = self.model.run(mol.smiles, topk=self.expansion_topk)
            if output_dict is not None:  # could be None if no reactions are possible
                reactants = output_dict["reactants"]
                scores = output_dict["scores"]
                templates = output_dict["template"]
                if len(reactants) > 0:
                    priors = np.clip(
                        np.asarray(scores), 1e-3, 1.0
                    )  # done by original paper
                    costs = -np.log(priors)

                    for j in range(len(reactants)):
                        rxn = BackwardReaction(
                            reactants=frozenset(
                                [Molecule(s) for s in reactants[j].split(".")]
                            ),
                            product=mol,
                            metadata=dict(
                                cost=float(costs[j]),
                                score=float(priors[j]),
                                template=templates[j],
                                logit=float(output_dict["logits"][j]),
                            ),
                        )
                        curr_output.append(rxn)

            output.append(curr_output)
        return output


class RetroStarReactionCostFunction(NoCacheNodeEvaluator):
    """Cost function designed to work with reaction model above."""

    def _evaluate_nodes(self, nodes, graph=None) -> list[float]:
        return [n.reaction.metadata["cost"] for n in nodes]
