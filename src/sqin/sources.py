"""Read-only: breakdown of best[] by origin (grid / pushdown depth / direct) with score mass. Usage: python -m sqin.sources"""
from collections import defaultdict

from . import store
from .construct import _score


def main():
    acc = defaultdict(lambda: [0, 0.0])
    for n in range(1, store.MAX_N + 1):
        _, L, src = store.load_best(n)
        s = str(src)
        root = s.split("<- ")[-1].split(" ")[0].split("(")[0]
        if s == "grid":
            kind = "grid"
        elif s.startswith("pushdown"):
            kind = f"pushdown<-{root} d={min(s.count('pushdown('), 5)}"
        else:
            kind = f"direct:{root}"
        g = "<=100" if n <= 100 else ("101-400" if n <= 400 else "401-1000")
        a = acc[(kind, g)]
        a[0] += 1
        a[1] += _score(float(L), n)
    for (kind, g), (c, m) in sorted(acc.items(), key=lambda x: -x[1][1]):
        print(f"{kind:44s} {g:9s} n={c:4d} mass={m:.4f}")


if __name__ == "__main__":
    main()
