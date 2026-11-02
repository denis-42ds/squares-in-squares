import math


def triv(n):
    return math.ceil(math.sqrt(n)) / math.sqrt(n) - 1.0

def k_of(n):
    return math.isqrt(n - 1) + 1

print(f"trivial total: {sum(triv(n) for n in range(1, 1001))}")
free = [n for n in range(1, 1001) if k_of(n) ** 2 - n <= 2]
print("perfect squares:", sum(1 for n in range(1, 1001) if k_of(n) ** 2 == n))
for lo, hi in [(1, 100), (101, 400), (401, 1000)]:
    r = range(lo, hi + 1)
    f = [n for n in free if lo <= n <= hi]
    t_all = sum(triv(n) for n in r)
    t_free = sum(triv(n) for n in f)
    print(f"{lo}-{hi} trivial {t_all:.6f} | free(k2,k2-1,k2-2): {len(f)} puzzles, {t_free:.6f}")
