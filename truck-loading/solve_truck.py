import warnings
from typing import Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from ortools.algorithms.python import knapsack_solver
from ortools.sat.python import cp_model

DEBUG = True
VERB = False

# Instance - Need to ensure all elements can fit in the bin, else the solution
# will be infeasible
# container: 0->length, 1->width, 2->max_weight, 3->cost
containers = [(40, 15, 20, 10)]

# Boxes:
#   "item_id":
#       Unique id of the item
#   "dim":
#       [0]: length (x dim)
#       [1]: width (y dim)
#       [2]: weight
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
    """
    Solve an instance of the truck loading problem.
    The objective is to pack a given set of items inside a number of trucks,
    while minimizing the total trucks cost.
    """

    def __init__(self):
        pass

    # OLD ---------------------------------------------------------------------+
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
        # IDEA: consider the items areas and limit it to vehicle's area
        weights = [
            [it["dim"][2] for it in items],
            [it["dim"][0] * it["dim"][1] for it in items],
        ]
        values = [1] * len(items)  # This solution maximizes the number of items

        # Use OR Tools' knapsack solver
        solver = knapsack_solver.KnapsackSolver(
            knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
            "KnapsackExample",
        )

        solver.init(values, weights, [truck[2], truck[0] * truck[1]])
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

        # for i in range(len(boxes)):
        #     model.Add(x_vars[i] == 0).OnlyEnforceIf(c_vars[i].Not())
        #     model.Add(x_vars[i] >= 0).OnlyEnforceIf(c_vars[i])
        #     model.Add(y_vars[i] == 0).OnlyEnforceIf(c_vars[i].Not())
        #     model.Add(y_vars[i] >= 0).OnlyEnforceIf(c_vars[i])
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

        # Add weight constraint
        model.Add(
            sum(c_vars[i] * boxes[i]["dim"][2] for i in range(len(boxes)))
            <= container[2]
        )

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
        available_boxes = []
        for i, box in enumerate(boxes):
            sol_curr_truck["item_id"].append(box["item_id"])
            sol_curr_truck["positions"].append(
                (solver.Value(x_vars[i]), solver.Value(y_vars[i]))
            )
            # Remove boxes used in current solution
            if solver.Value(c_vars[i]) == 0:
                available_boxes.append(box)

        return sol_curr_truck, available_boxes

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

    def solve_old(self, items: List[Dict], trucks: List):
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

            # usable, self.available_items = self.selectItemsSubset(
            #     next_truck, self.available_items
            # )

            sol_curr_truck, self.available_items = self.pack(
                next_truck, self.available_items
            )

    # -------------------------------------------------------------------------+
    # NEW ---------------------------------------------------------------------+

    def solve(
        self, items: List[Dict], trucks: List, max_truck_n: List[int] = []
    ):
        """
        Solve the problem using OR Tools' CP solver.

        Args:
            items: list of dict elements consisting of the items to be placed
            in the truck.
            trucks: list of available trucks (tuples, with elements: length,
            width, max_weight, cost)
            max_truck_n: list containing, for each truck type, the maximum units
            available. This variable is optional (default []), and it is meant
            to speed up the computation by reducing the variable dimensionality.
        """
        self.items = items
        self.trucks = trucks
        n_items = len(self.items)
        n_trucks = len(self.trucks)

        if max_truck_n == []:
            # Default value for the max. allowed number of trucks for each type
            # is n_items (1 item per truck)
            self.max_truck_n = [n_items] * n_trucks
        else:
            if len(max_truck_n) != n_trucks:
                raise ValueError(
                    "The maximum number of trucks per type should be a list with as many elements as trucks"
                )
            self.max_truck_n = max_truck_n

        self.model = cp_model.CpModel()

        # The maximum number of trucks that can be used for each type is given
        # by the elements of 'self.max_truck_n'

        # VARIABLES DEFINITION:
        # t_jk = 1 if truck k of type j is used (k is self.max_truck_n[j])
        t_vars = []
        for j in range(n_trucks):
            t_vars.append(
                [
                    self.model.NewBoolVar(name=f"t_{j},{k}")
                    for k in range(self.max_truck_n[j])
                ]
            )

        c_vars = []
        x_vars = []
        y_vars = []
        x_interval_vars = []
        y_interval_vars = []
        for i in range(n_items):
            c_vars.append([])
            x_vars.append([])
            y_vars.append([])
            x_interval_vars.append([])
            y_interval_vars.append([])
            for j in range(n_trucks):
                # i: item index
                # j: truck (type) index

                # c_ijk: 1 if item i is in k-th truck of type j
                c_vars[i].append(
                    [
                        self.model.NewBoolVar(name=f"c_({i},{j},{k})")
                        for k in range(self.max_truck_n[j])
                    ]
                )
                # x_ijk: x coordinate of the origin of item i in k-th truck j
                x_vars[i].append(
                    [
                        self.model.NewIntVar(
                            0,
                            trucks[j][0] - boxes[i]["dim"][0],
                            name=f"x_({i},{j},{k})",
                        )
                        for k in range(self.max_truck_n[j])
                    ]
                )
                # y_ijk: y coordinate of the origin of item i in k-th truck j
                y_vars[i].append(
                    [
                        self.model.NewIntVar(
                            0,
                            trucks[j][1] - boxes[i]["dim"][1],
                            name=f"y_({i},{j},{k})",
                        )
                        for k in range(self.max_truck_n[j])
                    ]
                )
                # Interval vars definition (x and y)
                x_interval_vars[i].append(
                    [
                        self.model.NewOptionalIntervalVar(
                            start=x_vars[i][j][k],
                            size=boxes[i]["dim"][0],
                            end=x_vars[i][j][k] + boxes[i]["dim"][0],
                            is_present=c_vars[i][j][k],
                            name=f"x_interval_({i},{j},{k})",
                        )
                        for k in range(self.max_truck_n[j])
                    ]
                )
                y_interval_vars[i].append(
                    [
                        self.model.NewOptionalIntervalVar(
                            start=y_vars[i][j][k],
                            size=boxes[i]["dim"][1],
                            end=y_vars[i][j][k] + boxes[i]["dim"][1],
                            is_present=c_vars[i][j][k],
                            name=f"y_interval_({i},{j},{k})",
                        )
                        for k in range(self.max_truck_n[j])
                    ]
                )

        # CONSTRAINTS DEFINITION
        # Each element should appear exactly 1 time (in 1 truck)
        for i in range(n_items):
            self.model.Add(
                sum(
                    [
                        c_vars[i][j][k]
                        for j in range(n_trucks)
                        for k in range(self.max_truck_n[j])
                    ]
                )
                == 1
            )

        objective = 0
        for j in range(n_trucks):
            for k in range(self.max_truck_n[j]):
                # Big-M constraint on the number of items in each truck - if the
                # truck is not considered, no item can be placed inside it
                self.model.Add(
                    sum([c_vars[i][j][k] for i in range(n_items)])
                    <= t_vars[j][k]
                ).OnlyEnforceIf(t_vars[j][k].Not())

                x_interval_vars_jk = [x[j][k] for x in x_interval_vars]
                y_interval_vars_jk = [y[j][k] for y in y_interval_vars]
                self.model.AddNoOverlap2D(
                    x_interval_vars_jk, y_interval_vars_jk
                )

                # Weight constraint
                self.model.Add(
                    sum(
                        c_vars[i][j][k] * boxes[i]["dim"][2]
                        for i in range(n_items)
                    )
                    <= trucks[j][2]
                )

                # OBJECTIVE FUNCTION: total trucks cost
                objective += t_vars[j][k] * trucks[j][3]

        self.model.Minimize(obj=objective)
        # Solve!
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 300.0
        self.solver.parameters.log_search_progress = True
        self.solver.log_callback = print
        # Enumerate all solutions.
        # solver.parameters.enumerate_all_solutions = True
        # Solve
        status = self.solver.Solve(self.model)
        print("+--------------------------------------------+")
        if status != cp_model.INFEASIBLE and status != cp_model.MODEL_INVALID:
            self.sol_found = True
            if status == cp_model.OPTIMAL:
                print("-> Optimal solution was found!")
            elif status == cp_model.FEASIBLE:
                print("-> Feasible solution found!")
            else:
                warnings.warn("Unknown solution status!")
        else:
            raise RuntimeError("No solution was found!")

        self.obj_val = self.solver.ObjectiveValue()
        print(f"> Objective Value (truck cost): {self.obj_val}")
        print("")

        # Display the solution (each truck k of type j)
        if VERB:
            for j in range(n_trucks):
                for k in range(self.max_truck_n[j]):
                    # Print
                    print(f"Truck {k + 1}, type {j + 1}:")
                    print(
                        f"> Number of items: {sum([self.solver.Value(c_vars[i][j][k]) for i in range(n_items)])}"
                    )
                    curr_tot_weight = sum(
                        [
                            boxes[i]["dim"][2]
                            for i in range(n_items)
                            if self.solver.Value(c_vars[i][j][k]) > 0
                        ]
                    )
                    print(f"> Total weight: {curr_tot_weight}")
            print("")

        self.used_trucks_sol = self.printSolution(
            boxes, trucks, c_vars, x_vars, y_vars
        )
        print(f"> {self.used_trucks_sol} trucks have been used")
        print("+--------------------------------------------+")

    def printSolution(self, boxes, trucks, c_vars, x_vars, y_vars) -> int:
        """
        Print the solution - only displaying the trucks that contain at least
        one item.
        The function returns the number of used trucks.

        Args:
            boxes: list of items (dict)
            trucks: list of truck types (dict)
            c_vars: variable 'c' from the model
            x_vars: variable 'x' from the model
            y_vars: variable 'y' from the model

        Returns:
            Integer number of trucks used (i.e., containing >0 elements); if no
            solution has been found yet, the returned value is -1.
        """
        if self.sol_found:
            n_used_trucks = 0
            for j in range(len(trucks)):
                for k in range(self.max_truck_n[j]):
                    # Check curr. truck contains at least 1 element
                    if sum(
                        [
                            self.solver.Value(c_vars[i][j][k])
                            for i in range(len(boxes))
                        ]
                    ):
                        n_used_trucks += 1
                        fig, ax = plt.subplots(1)
                        ax.set_xlim(0, trucks[j][0])
                        ax.set_ylim(0, trucks[j][1])
                        for i in range(len(boxes)):
                            if self.solver.Value(c_vars[i][j][k]) > 0:
                                ax.add_patch(
                                    patches.Rectangle(
                                        (
                                            self.solver.Value(x_vars[i][j][k]),
                                            self.solver.Value(y_vars[i][j][k]),
                                        ),
                                        boxes[i]["dim"][0],
                                        boxes[i]["dim"][1],
                                        facecolor="blue",
                                        alpha=0.2,
                                        edgecolor="b",
                                    )
                                )
                        # uniform axis
                        ax.set_aspect("equal", adjustable="box")
                        ax.set_title(f"Truck {j + 1} number {k + 1}")
                        fig.tight_layout()
                        plt.show()
            return n_used_trucks
        else:
            warnings.warn(
                "No solution was found yet! Please run the model first"
            )
            return -1


if __name__ == "__main__":
    # IDEA: decide beforehand which are the items to be considered among the
    # (long) list of ones that are provided - choose this by solving the
    # knapsack problem on the weight and on dimensions -> 3D knapsack

    truck_loading = TruckLoading()
    truck_loading.solve(boxes, containers)
