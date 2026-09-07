# intersection.py
"""
Given head lists from select_heads_p.py, computes the intersection of the
IP-derived and EP-derived head sets -- the "shared lexical task heads"
result (Table 1 / Fig. 3, one task's square).

Default ('single'): reproduces the primary result -- ONE canonical IP
condition (best-accuracy instruction template) intersected with ONE
canonical EP condition (fixed 5-shot).
"""
import argparse
import pickle
from functools import reduce


def load_head_list(pkl_path: str) -> list[tuple[int, int]]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["head_list"]  # matches select_heads_p.py's save format


def intersect_head_lists(*head_lists) -> list[tuple[int, int]]:
    sets = [set(hl) for hl in head_lists]
    return sorted(reduce(lambda a, b: a & b, sets))


def union_head_lists(*head_lists) -> list[tuple[int, int]]:
    sets = [set(hl) for hl in head_lists]
    return sorted(reduce(lambda a, b: a | b, sets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip_head_list_path", type=str, required=True,
        help="select_heads_p.py output for the best-accuracy IP template")
    parser.add_argument("--ep_head_list_paths", type=str, nargs="+", required=True,
        help="select_heads_p.py output(s) for EP. Pass ONE path (fixed 5-shot) "
             "for the primary Table 1/Fig. 3 result; pass multiple only with "
             "--ep_aggregation set to run the shot-count sensitivity check instead.")
    parser.add_argument("--ep_aggregation", type=str, default="single",
        choices=["single", "intersection", "union"],
        help="'single' = primary result, requires exactly one EP path. "
             "'intersection'/'union' = secondary robustness check across multiple EP paths.")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    ip_heads = load_head_list(args.ip_head_list_path)

    if args.ep_aggregation == "single":
        assert len(args.ep_head_list_paths) == 1, (
            "single aggregation expects exactly one EP path -- "
            "pass --ep_aggregation intersection/union for a multi-shot-count check"
        )
        ep_heads = load_head_list(args.ep_head_list_paths[0])
    else:
        ep_lists = [load_head_list(p) for p in args.ep_head_list_paths]
        ep_heads = (
            intersect_head_lists(*ep_lists) if args.ep_aggregation == "intersection"
            else union_head_lists(*ep_lists)
        )
        print(f"[secondary analysis] aggregated {len(args.ep_head_list_paths)} EP conditions "
              f"via '{args.ep_aggregation}' -> {len(ep_heads)} EP heads")

    shared = intersect_head_lists(ip_heads, ep_heads)
    print(f"IP heads: {len(ip_heads)}, EP heads: {len(ep_heads)}, Shared: {len(shared)}")
    print(shared)

    if args.save_path:
        with open(args.save_path, "wb") as f:
            pickle.dump({"shared_heads": shared, "ip_heads": ip_heads, "ep_heads": ep_heads}, f)
        print(f"Saved to {args.save_path}")