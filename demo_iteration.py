"""双频包裹相位联合确定绝对周数 —— 固定输入演示（离线、确定性）。

运行：python demo_iteration.py

内容（全部真实计算，无预录、无 sleep）：
1. 正常功能：互素频率 2 与 3（合成周期 2*pi），6x4 斜坡相位场，
   绝对区间跨约 3 个合成周期 —— 输出每个像素的全部可行区间与
   造成歧义的候选数，绝不只挑最接近 0 的一个。
2. 掩码：被掩码像素不参与任何空间传播，结果中为 None。
3. 边界情况：误差带恰接触（端点可行）与不相容观测（空解 +
   strict 模式抛 IncompatibleObservationsError）。
"""

import math
import time

from phasecut import (
    IncompatibleObservationsError,
    resolve_absolute_cycles,
    wrap_phase,
    TWO_PI,
)

PI = math.pi
F1, F2 = 2, 3  # 互素；合成周期 2*pi


def make_field(true_phi):
    p1 = [[wrap_phase(F1 * p) for p in row] for row in true_phi]
    p2 = [[wrap_phase(F2 * p) for p in row] for row in true_phi]
    return p1, p2


def main():
    t0 = time.perf_counter()

    print("=" * 68)
    print("1. 双频联合定周：频率 %d 与 %d，合成周期 2*pi = %.4f rad"
          % (F1, F2, TWO_PI))
    print("=" * 68)
    # 6x4 斜坡真值场，绝对相位落在 [-1, 2]；搜索区间 [-10, 10]
    # 跨约 3 个合成周期 -> 每像素应得 3 个候选。
    true_phi = [[-1.0 + 0.5 * c + 0.25 * r for c in range(4)] for r in range(6)]
    p1, p2 = make_field(true_phi)
    mask = [[True] * 4 for _ in range(6)]
    mask[2][1] = False  # 掩码一个像素：不参与、不传播
    res = resolve_absolute_cycles(
        p1, p2, F1, F2, err1=0.02, err2=0.03,
        phi_min=-10.0, phi_max=10.0, mask=mask,
    )
    print("区间 [-10, 10] 跨 %.2f 个合成周期；每像素候选数："
          % (20.0 / TWO_PI))
    for r in range(6):
        row = []
        for c in range(4):
            n = res.candidate_counts[r][c]
            row.append("  - " if n is None else " %d  " % n)
        print("  " + " ".join(row))
    print("歧义像素数 ambiguous_pixels =", res.ambiguous_pixels)
    print("空解像素数 empty_pixels     =", res.empty_pixels)
    print("掩码像素 (2,1) 的 intervals =", res.intervals[2][1])
    print("像素 (0,0)（真值 %.2f）的可行区间：" % true_phi[0][0])
    for lo, hi in res.intervals[0][0]:
        print("  [%.4f, %.4f]  中心 %.4f" % (lo, hi, (lo + hi) / 2))
    ok = all(
        iv is None or any(lo <= true_phi[r][c] <= hi for lo, hi in iv)
        for r in range(6) for c in range(4)
        for iv in [res.intervals[r][c]]
    )
    print("全部有效像素的真值都落在某个可行区间内:", ok)

    print()
    print("=" * 68)
    print("2. 边界情况：误差带恰接触（端点可行）")
    print("=" * 68)
    # 频率1带中心 1.0 半宽 0.5，频率2带中心 2.0 半宽 0.5，
    # 两带恰在 phi=1.5 接触 -> 唯一可行点（含端点）。
    w1 = wrap_phase(F1 * 1.0)
    w2 = wrap_phase(F2 * 2.0)
    res2 = resolve_absolute_cycles(
        [[w1]], [[w2]], F1, F2, err1=1.0, err2=1.5,
        phi_min=0.0, phi_max=3.0,
    )
    (lo, hi), = res2.intervals[0][0]
    print("恰接触的可行区间: [%.12f, %.12f]（退化为接触点 1.5）" % (lo, hi))

    print()
    print("=" * 68)
    print("3. 边界情况：不相容观测")
    print("=" * 68)
    # 频率1说 phi=0、频率2说 phi=1，误差带很小 -> 无可行解。
    bad1 = wrap_phase(F1 * 0.0)
    bad2 = wrap_phase(F2 * 1.0)
    res3 = resolve_absolute_cycles(
        [[bad1]], [[bad2]], F1, F2, err1=0.05, err2=0.05,
        phi_min=-PI, phi_max=PI,
    )
    print("默认模式：intervals =", res3.intervals[0][0],
          " empty_pixels =", res3.empty_pixels)
    try:
        resolve_absolute_cycles(
            [[bad1]], [[bad2]], F1, F2, err1=0.05, err2=0.05,
            phi_min=-PI, phi_max=PI, strict=True,
        )
    except IncompatibleObservationsError as exc:
        print("strict 模式实际抛出:", type(exc).__name__)
        print("  ", exc)

    dt = time.perf_counter() - t0
    print()
    print("全部演示真实计算完成，用时 %.3f 秒。" % dt)


if __name__ == "__main__":
    main()
