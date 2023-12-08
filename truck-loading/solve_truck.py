from typing import Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from ortools.algorithms.python import knapsack_solver
from ortools.sat.python import cp_model

DEBUG = True

# Instance - Need to ensure all elements can fit in the bin, else the solution
# will be infeasible
container = (40, 15, 40)

# Boxes:
#   [0]: length (x dim)
#   [1]: width (y dim)
#   [2]: weight
boxes = [
    {"item_id": "I0001", "dim": (11, 3, 4)},
    {"item_id": "I0002", "dim": (13, 3, 2)},
    {"item_id": "I0003", "dim": (9, 2, 4)},
    {"item_id": "I0004", "dim": (7, 2, 6)},
    {"item_id": "I0005", "dim": (9, 3, 1)},
    {"item_id": "I0006", "dim": (7, 3, 3)},
    {"item_id": "I0007", "dim": (11, 2, 3)},
    {"item_id": "I0008", "dim": (13, 2, 2)},
    {"item_id": "I0009", "dim": (11, 4, 7)},
    {"item_id": "I0010", "dim": (13, 4, 6)},
    {"item_id": "I0010", "dim": (3, 5, 1)},
    {"item_id": "I0012", "dim": (11, 2, 2)},
    {"item_id": "I0013", "dim": (2, 2, 3)},
    {"item_id": "I0014", "dim": (11, 3, 4)},
    {"item_id": "I0015", "dim": (2, 3, 5)},
    {"item_id": "I0016", "dim": (5, 4, 5)},
    {"item_id": "I0017", "dim": (6, 4, 4)},
    {"item_id": "I0018", "dim": (12, 2, 3)},
    {"item_id": "I0019", "dim": (1, 2, 2)},
    {"item_id": "I0020", "dim": (3, 5, 3)},
    {"item_id": "I0021", "dim": (13, 5, 1)},
    {"item_id": "I0022", "dim": (12, 4, 4)},
    {"item_id": "I0023", "dim": (1, 4, 1)},
    {"item_id": "I0024", "dim": (5, 2, 3)},
    {"item_id": "I0025", "dim": (6, 2, 1)},  # add to make tight
    {"item_id": "I0026", "dim": (6, 3, 1)},  # add to make infeasible
]


class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solution_limit = limit

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            print(f"{v}={self.Value(v)}", end=" ")
        print()
        if self.__solution_count >= self.__solution_limit:
            print(f"Stop search after {self.__solution_limit} solutions")
            self.StopSearch()

    def solution_count(self):
        return self.__solution_count


class TruckLoading:
    def __init__(self):
        pass

    def selectItemsSubset(
        self, truck: Tuple[int, int, int], items: List[Dict]
    ) -> Tuple[List, List]:
        """
        Given a the complete set of items, select the subset that will be passed
        to the `pack` function, to solve the container in the optimal way.

        The idea is to maximize truck occupation (TODO: decide if in terms of
        object count or surface/weight occupancy).

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
        weights = [[it["dim"][2] for it in items]]
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
        ), f"[selectItemsSubset]: items don't add up \
({len(items)} vs. {len(selected_items) + len(discarded_items)})"

        return selected_items, discarded_items

    def pack(
        self,
        container: Tuple[int, int, int],
        boxes: List[Dict],
        plot: bool = True,
    ):
        """
        Solve the 2D bin packing problem on the container using the boxes.

        NOTE: the solution will be feasible only if all boxes can actually fit
        inside the container (both in terms of dimensions and weight.

        Args:
            - container: bin where the objects should be fit; elements 0 and 1
            are the x and y dimensions, while element 2 is the maximum weight
            - boxes: list of items that have to be placed inside the container;
            they are tuples of length 3 (elements 0 and 1: dimensions, element
            2: weight)

        Output:
            - None
        """
        # Define the CP-sat model
        model = cp_model.CpModel()

        # Create a bool variable c_i indicating if item i is considered
        c_vars = [model.NewBoolVar(f"c_{i}") for i in range(len(boxes))]

        # We have to create the variable for the bottom left corner of the
        # boxes.
        # We directly limit their range, such that the boxes are inside the
        # container
        x_vars = [
            model.NewIntVar(0, container[0] - box["dim"][0], name=f"x1_{i}")
            for i, box in enumerate(boxes)
        ]
        y_vars = [
            model.NewIntVar(0, container[1] - box["dim"][1], name=f"y1_{i}")
            for i, box in enumerate(boxes)
        ]

        for i in range(len(boxes)):
            model.Add(x_vars[i] == 0).OnlyEnforceIf(c_vars[i].Not())
            model.Add(x_vars[i] >= 0).OnlyEnforceIf(c_vars[i])
            model.Add(y_vars[i] == 0).OnlyEnforceIf(c_vars[i].Not())
            model.Add(y_vars[i] >= 0).OnlyEnforceIf(c_vars[i])
        # Interval variables are actually more like constraint containers, that
        # are then passed to the no overlap constraint.
        # Note that we could also make size and end variables, but we don't need
        # them here
        x_interval_vars = [
            model.NewOptionalIntervalVar(
                start=x_vars[i],
                size=box["dim"][0],
                end=x_vars[i] + box["dim"][0],
                is_present=c_vars[i],
                name=f"x_interval_{i}",
            )
            for i, box in enumerate(boxes)
        ]
        y_interval_vars = [
            model.NewOptionalIntervalVar(
                start=y_vars[i],
                size=box["dim"][1],
                end=y_vars[i] + box["dim"][1],
                is_present=c_vars[i],
                name=f"y_interval_{i}",
            )
            for i, box in enumerate(boxes)
        ]

        # Enforce that no two rectangles overlap
        model.AddNoOverlap2D(x_interval_vars, y_interval_vars)

        # Add objective function - it pushes all items towards (x, y) = (0, 0)
        model.Maximize(sum([c_vars[i] for i in range(len(x_vars))]))

        # Solve!
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 300.0
        solver.parameters.log_search_progress = True
        solver.log_callback = print
        # Enumerate all solutions.
        # solver.parameters.enumerate_all_solutions = True
        # Solve.
        status = solver.Solve(model)
        assert status == cp_model.OPTIMAL

        if plot:
            # plot the solution
            fig, ax = plt.subplots(1)
            ax.set_xlim(0, container[0])
            ax.set_ylim(0, container[1])
            for i, box in enumerate(boxes):
                if solver.Value(c_vars[i]) > 0:
                    ax.add_patch(
                        patches.Rectangle(
                            (solver.Value(x_vars[i]), solver.Value(y_vars[i])),
                            box["dim"][0],
                            box["dim"][1],
                            facecolor="blue",
                            alpha=0.2,
                            edgecolor="b",
                        )
                    )
            # uniform axis
            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout()
            plt.show()

        # Store the solution:
        sol_curr_truck = {"positions": [], "item_id": []}

        for i, box in enumerate(boxes):
            sol_curr_truck["item_id"].append(box["item_id"])
            sol_curr_truck["positions"].append(
                (solver.Value(x_vars[i]), solver.Value(y_vars[i]))
            )

        return sol_curr_truck

    def selectTruck(self):
        """
        Select the truck to be used for the next solution.

        """
        if self.trucks is None:
            raise ValueError("No trucks list has been provided yet!")

        if len(self.trucks) == 1:
            return self.trucks[0]
        else:
            # TODO: implement - HOW??
            raise NotImplementedError(
                "This function only supports a single truck type!"
            )

    def solve(self, items: List[Dict], trucks: List):
        """
        Solve the truck loading problem for the set of items and trucks
        provided.
        The goal is to minimize the number of trucks used.

        Args:
            items: list of items (Dict: item_id, dim: (x, y, weight))
            trucks: list of available trucks (x, y, max_weight)
        """
        self.items = items
        self.trucks = trucks

        self.available_items = items

        # TODO: decide how to assign items to trucks (add different trucks)

        while len(self.available_items) > 0:
            # Select the truck
            next_truck = self.selectTruck()

            usable, self.available_items = self.selectItemsSubset(
                next_truck, self.available_items
            )

            self.pack(next_truck, usable)


if __name__ == "__main__":
    # IDEA: decide beforehand which are the items to be considered among the
    # (long) list of ones that are provided - choose this by solving the
    # knapsack problem on the weight and on dimensions -> 3D knapsack

    truck_loading = TruckLoading()
    truck_loading.solve(boxes, [container])
