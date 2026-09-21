from __future__ import annotations


def optimal_temporal_matching(
    predicted: list[int], gold: list[int], tolerance: int
) -> list[tuple[int, int]]:
    """Match ordered temporal events by cardinality, then total offset.

    Contacts are ordered in time, so dynamic programming gives the exact
    maximum-cardinality matching without the greedy counterexample where a
    close early pair blocks two valid pairs later.
    """
    predicted = [int(value) for value in predicted]
    gold = [int(value) for value in gold]
    n, m = len(predicted), len(gold)
    if not n or not m:
        return []

    # Each cell stores (number of matches, negative total offset, pairs).
    dp: list[list[tuple[int, int, tuple[tuple[int, int], ...]]]] = [
        [(0, 0, ()) for _ in range(m + 1)] for _ in range(n + 1)
    ]

    def better(left: tuple[int, int, tuple[tuple[int, int], ...]], right: tuple[int, int, tuple[tuple[int, int], ...]]):
        return left if (left[0], left[1]) >= (right[0], right[1]) else right

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = better(dp[i - 1][j], dp[i][j - 1])
            distance = abs(predicted[i - 1] - gold[j - 1])
            if distance <= tolerance:
                previous = dp[i - 1][j - 1]
                candidate = (
                    previous[0] + 1,
                    previous[1] - distance,
                    previous[2] + ((i - 1, j - 1),),
                )
                best = better(best, candidate)
            dp[i][j] = best
    return list(dp[n][m][2])
