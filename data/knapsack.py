from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver
import numpy as np


# https://en.wikipedia.org/wiki/Knapsack_problem
# 0-1 knapsack problem
#   dynamic programming solution for the 0-1 knapsack problem
def knapsack_dp(values, weights, capacity):
    check_inputs(values, weights, capacity)
    n_items = len(values)
    cap_range = range(capacity + 1)
    item_range = range(n_items + 1)

    # The table[i][w] will have the maximum value that can be attained
    # with weight less than or equal to w using items up to i (first i items).
    table = [[0 for _ in cap_range] for _ in item_range]

    for i in item_range[1:]:    # 1...n_items
        wi = weights[i - 1]
        vi = values[i - 1]
        for w in cap_range:     # 0...capacity
            prev_max = table[i - 1][w]
            if w < wi:  # the new item is more than the current weight limit
                table[i][w] = prev_max
            else:       # the new item can be added
                # choose the better between the max values:
                #   1) vi + table[i - 1][w - wi] when the new item is added,
                #      where table[i - 1][w - wi] is the previous max
                #      considering room for the new item (w - wi)
                #   2) prev_max when the new item is not added
                table[i][w] = max(vi + table[i - 1][w - wi], prev_max)

    selected = []
    w = capacity
    for i in range(n_items, 0, -1):  # n_items...1
        item_added = table[i][w] != table[i - 1][w]
        if item_added:
            selected.append(i - 1)
            w -= weights[i - 1]
    selected.sort()

    return selected


solver = KnapsackSolver(KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
                        "KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER")


def knapsack_ortools(values, weights, capacity):
    check_inputs(values, weights, capacity)
    scale = 1e8
    values = (np.asarray(values) * scale).astype(int).tolist()
    weights = [weights]
    capacities = [capacity]

    solver.Init(values, weights, capacities)
    _ = solver.Solve()

    n_items = len(values)
    selected = [i for i in range(n_items) if solver.BestSolutionContains(i)]
    return selected


def check_inputs(values, weights, capacity):
    # check variable type
    assert(isinstance(values, list))
    assert(isinstance(weights, list))
    assert(isinstance(capacity, int))

    assert(len(values) == len(weights))

    # check value type
    assert(all(isinstance(val, int) or isinstance(val, float) for val in values))
    assert(all(isinstance(val, int) for val in weights))

    # check validity of value
    assert(all(0 <= val for val in weights))
    assert(0 < capacity)


knapsack = knapsack_ortools


if __name__ == "__main__":
    values = [5, 4, 3, 2]
    weights = [4, 3, 2, 1]
    capacity = 6
    selected = knapsack(values, weights, capacity)
    print(selected)
    #     | w
    #     | 0   1   2   3   4   5   6
    # --------------------------------
    # i 0 | 0   0   0   0   0   0   0
    #   1 | 0   0   0   0   5   5   5
    #   2 | 0   0   0   4<- 5   5   5
    #   3 | 0   0   3   4   5   7<- 8
    #   4 | 0   2   3   5   6   7   9<-
    assert(selected == [1, 2, 3])
