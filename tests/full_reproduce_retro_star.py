"""Try to fully reproduce the results of retro-star with their pre-trained model."""
from __future__ import annotations

import argparse
import logging
import math
import sys
import numpy as np

from tqdm.auto import tqdm

from syntheseus.search.chem import Molecule
from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator

from retro_star_task import (
    RetroStarReactionModel,
    RetroStarReactionCostFunction,
    RetroStarInventory,
    RetroStarValueMLP,
    test_molecules,
)


def retro_star_search(
    smiles_list: list[str],
    use_value_function: bool,
    use_tqdm: bool = False,
    limit_rxn_model_calls: int = 500,
) -> list[float]:
    """

    Do search on a list of SMILES strings and report the time of first solution.
    """

    # Initialize algorithm.
    rxn_model = RetroStarReactionModel(
        use_cache=False
    )  # no caching (original paper did not use caching)
    inventory = RetroStarInventory()
    if use_value_function:
        value_fn = RetroStarValueMLP()
    else:
        value_fn = ConstantNodeEvaluator(0.0)
    alg = RetroStarSearch(
        reaction_model=rxn_model,
        mol_inventory=inventory,
        limit_reaction_model_calls=limit_rxn_model_calls,
        and_node_cost_fn=RetroStarReactionCostFunction(),
        value_function=value_fn,
        time_limit_s=10_000,
        max_expansion_depth=50,  # prevent overly-deep solutions (note: not done in original paper)
        prevent_repeat_mol_in_trees=True,  # original paper did this
        unique_nodes=False,  # run on tree, not graph
    )

    # Do search
    logger = logging.getLogger("retro_star_reproduce")
    min_soln_times: list[float] = []
    if use_tqdm:
        smiles_iter = tqdm(smiles_list)
    else:
        smiles_iter = smiles_list
    for i, smiles in enumerate(smiles_iter):
        logger.debug(f"Start search {i}/{len(smiles_list)}. SMILES: {smiles}")
        alg.reset()
        output_graph, _ = alg.run_from_mol(Molecule(smiles))
        for node in output_graph.nodes():
            node.data["analysis_time"] = node.data["num_calls_rxn_model"]
        soln_time = get_first_solution_time(output_graph)
        assert (
            math.isinf(soln_time) != output_graph.root_node.has_solution
        )  # sanity check
        logger.debug(f"Done: nodes={len(output_graph)}, solution time = {soln_time}")
        min_soln_times.append(soln_time)

    return min_soln_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit_num_smiles",
        type=int,
        default=None,
        help="Maximum number of SMILES to run.",
    )
    parser.add_argument(
        "--rxn_model_calls",
        type=int,
        default=500,
        help="Allowed number of calls to reaction model.",
    )
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        filemode="w",
    )

    # Load all SMILES to test
    test_smiles = test_molecules.get_190_hard_test_smiles()
    if args.limit_num_smiles is not None:
        test_smiles = test_smiles[: args.limit_num_smiles]

    # Run without value function (retro*-0)
    rxn_model_budget = args.rxn_model_calls
    results_no_value_fn = np.asarray(
        retro_star_search(
            smiles_list=test_smiles,
            use_tqdm=True,
            limit_rxn_model_calls=rxn_model_budget,
            use_value_function=False,
        )
    )
    results_with_value_fn = np.asarray(
        retro_star_search(
            smiles_list=test_smiles,
            use_tqdm=True,
            limit_rxn_model_calls=rxn_model_budget,
            use_value_function=True,
        )
    )

    # Tabulate results (should closely match paper)
    for t in range(0, rxn_model_budget + 1, 50):
        print(
            f"t={t:>5d} "
            f"Retro*-0={np.average(results_no_value_fn<=t):.4f} "
            f"Retro*={np.average(results_with_value_fn<=t):.4f}"
        )
