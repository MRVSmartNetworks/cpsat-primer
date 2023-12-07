from typing import List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from ortools.algorithms.python import knapsack_solver
from ortools.sat.python import cp_model

DEBUG = True

# Instance - Need to ensure all elements can fit in the bin, else the solution
# will be infeasible
container = (40, 15, 20)

# Boxes:
#   [0]: length (x dim)
#   [1]: width (y dim)
#   [2]: weight
boxes = [
    (11, 3, 4),
    (13, 3, 2),
    (9, 2, 4),
    (7, 2, 6),
    (9, 3, 1),
    (7, 3, 3),
    (11, 2, 3),
    (13, 2, 2),
    (11, 4, 7),
    (13, 4, 6),
    (3, 5, 1),
    (11, 2, 2),
    (2, 2, 3),
    (11, 3, 4),
    (2, 3, 5),
    (5, 4, 5),
    (6, 4, 4),
    (12, 2, 3),
    (1, 2, 2),
    (3, 5, 3),
    (13, 5, 1),
    (12, 4, 4),
    (1, 4, 1),
    (5, 2, 3),
    (6, 2, 1),  # add to make tight
    (6, 3, 1),  # add to make infeasible
]


def selectItemsSubset(
    truck: Tuple[int, int, int], items: List[Tuple]
) -> Tuple[List, List]:
    """
    Given a the complete set of items, select the subset that will be passed to
    the `pack` function, to solve the container in the optimal way.

    The idea is to maximize truck occupation (TODO: decide if in terms of object
    count or surface/weight occupancy).

    Args:
        - truck: the bin - (x_dim, y_dim, weight)
        - items: list containing all items

    Output:
        - List containing the selected items
        - List containing the discarded items
    """
    selected_items = []
    discarded_items = []

    # Solve 3D knapsack on weight (TODO: and dimensions[?])
    weights = [[it[2] for it in items]]
    values = [1] * len(items)  # This solution maximizes the number of items

    # Use OR Tools' knapsack solver
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample",
    )

    solver.init(values, weights, [truck[2]])
    sol_value = solver.solve()
    if DEBUG:
        print(f"Solution: {sol_value}")

    # Build solution - assign the items
    for i in range(len(items)):
        if solver.best_solution_contains(i):
            selected_items.append(items[i])
        else:
            discarded_items.append(items[i])

    if DEBUG:
        print(f"Selected items: {selected_items}")

    assert len(items) == len(selected_items) + len(
        discarded_items
    ), f"[selectItemsSubset]: items don't add up ({len(items)} vs. {len(selected_items) + len(discarded_items)})"

    return selected_items, discarded_items


def pack(
    container: Tuple[int, int, int], boxes: List[Tuple], plot: bool = True
):
    """
    Solve the 2D bin packing problem on the container using the boxes.

    NOTE: the solution will be feasible only if all boxes can actually fit
    inside the container (both in terms of dimensions and weight.

    Args:
        - container: bin where the objects should be fit; elements 0 and 1 are
        the x and y dimensions, while element 2 is the maximum weight
        - boxes: list of items that have to be placed inside the container; they
        are tuples of length 3 (elements 0 and 1: dimensions, element 2: weight)

    Output:
        - None
    """
    # Define the CP-sat model
    model = cp_model.CpModel()

    # We have to create the variable for the bottom left corner of the boxes.
    # We directly limit their range, such that the boxes are inside the
    # container
    x_vars = [
        model.NewIntVar(0, container[0] - box[0], name=f"x1_{i}")
        for i, box in enumerate(boxes)
    ]
    y_vars = [
        model.NewIntVar(0, container[1] - box[1], name=f"y1_{i}")
        for i, box in enumerate(boxes)
    ]
    # Interval variables are actually more like constraint containers, that are
    # then passed to the no overlap constraint
    # Note that we could also make size and end variables, but we don't need
    # them here
    x_interval_vars = [
        model.NewIntervalVar(
            start=x_vars[i],
            size=box[0],
            end=x_vars[i] + box[0],
            name=f"x_interval_{i}",
        )
        for i, box in enumerate(boxes)
    ]
    y_interval_vars = [
        model.NewIntervalVar(
            start=y_vars[i],
            size=box[1],
            end=y_vars[i] + box[1],
            name=f"y_interval_{i}",
        )
        for i, box in enumerate(boxes)
    ]
    # Enforce that no two rectangles overlap
    model.AddNoOverlap2D(x_interval_vars, y_interval_vars)

    # Add objective function - it pushes all items towards (x, y) = (0, 0)
    # model.Minimize(sum([(x_vars[i] + y_vars[i]) for i in range(len(x_vars))]))

    # Solve!
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.log_callback = print
    status = solver.Solve(model)

    if plot:
        # plot the solution
        _, ax = plt.subplots(1)
        ax.set_xlim(0, container[0])
        ax.set_ylim(0, container[1])
        for i, box in enumerate(boxes):
            ax.add_patch(
                patches.Rectangle(
                    (solver.Value(x_vars[i]), solver.Value(y_vars[i])),
                    box[0],
                    box[1],
                    facecolor="blue",
                    alpha=0.2,
                    edgecolor="b",
                )
            )
        # uniform axis
        ax.set_aspect("equal", adjustable="box")
        plt.show()
    assert status == cp_model.OPTIMAL


if __name__ == "__main__":
    # IDEA: decide beforehand which are the items to be considered among the
    # (long) list of ones that are provided - choose this by solving the
    # knapsack problem on the weight and on dimensions -> 3D knapsack
    keep, discard = selectItemsSubset(container, boxes)

    print(f"Discarded items: {discard}")

    pack(container, keep)
