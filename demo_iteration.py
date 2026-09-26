"""demo_iteration — 双频绝对周数求解的真实计算演示（无预录、无 sleep）。

展示：
1. 唯一解：互素频率 3/4，区间小于一个合成周期；
2. 歧义：区间跨一个合成周期，候选数如实输出（不挑最接近 0 的）；
3. 边界情况：误差带恰接触仍可行、不相容观测如实报告空候选。
"""

import math
import time

from phasecut import solve_two_frequency, unwrap_phase, wrap_phase

TWO_PI = 2.0 * math.pi


def make_observations(phi_field, k1, k2):
    h = len(phi_field)
    w = len(phi_field[0])
    phase1 = [[wrap_phase(k1 * phi_field[r][c]) for c in range(w)] for r in range(h)]
    phase2 = [[wrap_phase(k2 * phi_field[r][c]) for c in range(w)] for r in range(h)]
    mask = [[True] * w for _ in range(h)]
    return phase1, phase2, mask


def main():
    t0 = time.perf_counter()

    print("=" * 64)
    print("1) 唯一解：频率 3/4（互素），Phi=1.0，区间 [-1, 5]（宽 6 < 2π）")
    print("=" * 64)
    field = [[1.0, 2.5], [0.5, 4.0]]
    p1, p2, mask = make_observations(field, 3, 4)
    res = solve_two_frequency(p1, p2, mask, 3, 4, 0.0, 0.0, -1.0, 5.0)
    for r in range(2):
        for c in range(2):
            cand = res.candidates[r][c][0]
            print(
                f"  像素({r},{c}): 真值 Phi={field[r][c]}  "
                f"解=[{cand.lo:.6f}, {cand.hi:.6f}] 周数 (n1,n2)=({cand.n1},{cand.n2})"
            )
    print(f"  统计: 唯一 {res.solved_pixels}，歧义 {res.ambiguous_pixels}，空 {res.empty_pixels}")

    print()
    print("=" * 64)
    print("2) 歧义：同一观测，区间 [-1, 2π+1] 跨一个合成周期（2π）")
    print("=" * 64)
    res = solve_two_frequency(p1, p2, mask, 3, 4, 0.0, 0.0, -1.0, TWO_PI + 1.0)
    for r in range(2):
        for c in range(2):
            cands = res.candidates[r][c]
            los = ", ".join(f"{cd.lo:.4f}" for cd in cands)
            print(f"  像素({r},{c}): 候选数={len(cands)}  lo=[{los}]")
    print(f"  统计: 唯一 {res.solved_pixels}，歧义 {res.ambiguous_pixels}，空 {res.empty_pixels}")
    print("  → 歧义候选全部如实列出，没有挑最接近 0 的那个。")

    print()
    print("=" * 64)
    print("3) 边界情况")
    print("=" * 64)
    # 3a. 误差带恰接触：k1=1, k2=2，观测1 给 [-1,1]，观测2 零误差钉 Phi=1.0。
    res = solve_two_frequency([[0.0]], [[2.0]], [[True]], 1, 2, 1.0, 0.0, -3.0, 3.0)
    cand = res.candidates[0][0][0]
    print(f"  3a) 误差带恰接触: 候选数=1，区间=[{cand.lo:.6f}, {cand.hi:.6f}]（闭区间，接触即可行）")

    # 3b. 不相容观测：w1=0（Phi≈0）与 w2=3.0（Phi≈1.5），误差 0.05。
    res = solve_two_frequency([[0.0]], [[3.0]], [[True]], 1, 2, 0.05, 0.05, -3.0, 3.0)
    print(f"  3b) 不相容观测: 候选数={res.counts[0][0]}，empty_pixels={res.empty_pixels}（如实报告，不抛异常）")

    # 3c. 旧单幅接口回归冒烟。
    phase = [[wrap_phase(0.1 * (r + c)) for c in range(4)] for r in range(4)]
    old = unwrap_phase(phase, [[True] * 4 for _ in range(4)])
    print(f"  3c) 旧接口 unwrap_phase 冒烟: 区域数={len(old.regions)}，unmet_edges={len(old.unmet_edges)}")

    dt = time.perf_counter() - t0
    print()
    print(f"全部真实计算完成，用时 {dt:.3f} 秒。")


if __name__ == "__main__":
    main()

