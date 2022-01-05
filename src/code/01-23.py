import numpy as np


def test_independence(a_arr, b_arr, n, seed=42):
    rng = np.random.default_rng(seed)
    roll = rng.integers(1, 7, n)
    a_count = np.count_nonzero(np.isin(roll, a_arr))
    b_count = np.count_nonzero(np.isin(roll, b_arr))
    ab_arr = list(set(a_arr).intersection(set(b_arr)))
    ab_count = np.count_nonzero(np.isin(roll, ab_arr))

    p_a = a_count / n
    p_b = b_count / n
    p_ab = ab_count / n

    print(f"A = {a_arr}, B = {b_arr}, AB = {ab_arr}")
    print(f"P(A) = {p_a}")
    print(f"P(B) = {p_b}")
    print(f"expected P(AB) if independent = {p_a*p_b:.3}")
    print(f"actual P(AB) = {p_ab}")


test_independence([2, 4, 6], [1, 2, 3, 4], 10000)
print()
test_independence([1, 2, 3, 4], [3, 4, 5, 6], 10000)
