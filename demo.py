"""phasecut 固定输入演示（离线、确定性、无第三方依赖）。

运行：python demo.py

内容：
1. 无涡旋斜坡（频繁跨 pi 包裹）：无残差、无切口、单区精确还原。
2. 单涡旋参考（解析 atan2 场）：对比“不加切口的朴素洪泛”与加切口后的
   环路误差——前者在涡核环留下约 2π 的矛盾，后者非切口边全部一致。
3. 实际触发的边界：NaN 输入被带坐标拒绝；孤岛涡旋证明无解；
   搜索预算耗尽与无解明确区分。
4. 相邻两帧全局整数周对齐 align_frames：跨 2π 时间跳变（含局部缺测，
   只用共同有效像素）正常对齐；精确半点并列取较小整数；无共同像素
   返回未对齐并保留掩码；两帧形状不一致给可定位错误（无半更新）。
"""

import math
import time

from phasecut import (
    InvalidPhaseInputError,
    NoSolutionError,
    SearchBudgetExhausted,
    TWO_PI,
    align_frames,
    unwrap_phase,
    wrap_phase,
)

PI = math.pi


def naive_flood_edge_residuals(wrapped, mask):
    """不设切口的朴素单源洪泛，独立计算每条有效边的残差。

    residual = (phi_b - phi_a) - wrap(w_b - w_a)。
    树边（首次到达边）必然为 0；涡旋存在时，闭合环上的非树边无法同时
    满足，会留下约 ±2π 的残差——这就是被“看似平滑”的结果掩盖的矛盾。
    返回 {(r1,c1,r2,c2): residual}（只存右向/下向正向边）。
    """
    h, wd = len(mask), len(mask[0])
    phi = [[None] * wd for _ in range(h)]
    start = next((r, c) for r in range(h) for c in range(wd) if mask[r][c])
    phi[start[0]][start[1]] = wrapped[start[0]][start[1]]
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < wd) or not mask[nr][nc]:
                continue
            if phi[nr][nc] is not None:
                continue
            d = wrap_phase(wrapped[nr][nc] - wrapped[r][c])
            expect = phi[r][c] + d
            k = int(math.floor(abs((expect - wrapped[nr][nc]) / TWO_PI) + 0.5))
            k = k if expect - wrapped[nr][nc] >= 0 else -k
            phi[nr][nc] = wrapped[nr][nc] + k * TWO_PI
            stack.append((nr, nc))

    residuals = {}
    for r in range(h):
        for c in range(wd):
            if not mask[r][c]:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr >= h or nc >= wd or not mask[nr][nc]:
                    continue
                residuals[(r, c, nr, nc)] = (
                    phi[nr][nc] - phi[r][c]
                    - wrap_phase(wrapped[nr][nc] - wrapped[r][c])
                )
    return residuals


def wrapped_loop_sum(wrapped, r, c):
    """直接由包裹输入独立计算环 (r,c) 的包裹邻差环和（涡核环约 ±2π）。"""
    return (
        wrap_phase(wrapped[r][c + 1] - wrapped[r][c])
        + wrap_phase(wrapped[r + 1][c + 1] - wrapped[r][c + 1])
        + wrap_phase(wrapped[r + 1][c] - wrapped[r + 1][c + 1])
        + wrap_phase(wrapped[r][c] - wrapped[r + 1][c])
    )


def cut_touches_plaquette(cut, r, c):
    corners = {(r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)}
    return (cut.row1, cut.col1) in corners and (cut.row2, cut.col2) in corners


def section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main():
    t0 = time.perf_counter()

    # --- 1. 无涡旋斜坡（跨 pi） -------------------------------------------
    section("1. 无涡旋斜坡 12x12（行梯度 0.9，列梯度 2.5，频繁跨 pi）")
    h = w = 12
    ar, ac = 0.9, 2.5
    truth = [[ar * r + ac * c for c in range(w)] for r in range(h)]
    wrapped = [[wrap_phase(v) for v in row] for row in truth]
    mask = [[True] * w for _ in range(h)]

    res = unwrap_phase(wrapped, mask)
    print(f"残差数            : {len(res.residues)}")
    print(f"切口边数          : {len(res.cuts)}")
    print(f"连通区数          : {len(res.regions)}（锚点 {res.regions[0].anchor}，"
          f"基准 k={res.regions[0].baseline}）")
    print(f"未满足边残差数    : {len(res.unmet_edges)}")
    anchor = res.unwrapped[0][0]
    worst = max(
        abs(res.unwrapped[r][c] - anchor - truth[r][c])
        for r in range(h) for c in range(w)
    )
    print(f"展开差值对真值梯度的最大偏差: {worst:.3e} rad（口径 1e-9）")
    print(f"角点示例: 包裹=({wrapped[0][11]:+.4f},{wrapped[11][11]:+.4f}) "
          f"展开=({res.unwrapped[0][11]:+.4f},{res.unwrapped[11][11]:+.4f})")

    # --- 2. 单涡旋参考：切口前后矛盾度量 ----------------------------------
    section("2. 单涡旋参考 9x9（解析 atan2 场，涡核在像素角点 (0.5,0.5)）")
    h = w = 9
    vortex = [
        [wrap_phase(math.atan2(r - 0.5, c - 0.5)) for c in range(w)]
        for r in range(h)
    ]
    mask = [[True] * w for _ in range(h)]

    # 度量一：直接由包裹输入算每个环的包裹邻差环和（涡核环必为 ±2π）。
    loop_sums = {
        (r, c): wrapped_loop_sum(vortex, r, c)
        for r in range(h - 1) for c in range(w - 1)
    }
    bad_loops = sorted((p, s) for p, s in loop_sums.items() if abs(s) > 1e-9)
    print("输入本身的矛盾（包裹邻差环和）:")
    for (r, c), s in bad_loops:
        print(f"  涡核环 @ (row={r}, col={c}): 环和 = {s:+.12f} "
              f"({s / TWO_PI:+.3f}·2π)")

    # 度量二：朴素单源洪泛（不设切口）后每条边的残差。
    before = naive_flood_edge_residuals(vortex, mask)
    bad_edges = sorted((k, v) for k, v in before.items() if abs(v) > 1e-9)
    print("切口前（朴素洪泛、不设切口，结果看似平滑）:")
    print(f"  残差超 {1e-9:g} 的边数 : {len(bad_edges)}")
    for (r1, c1, r2, c2), v in bad_edges:
        print(f"  矛盾边 ({r1},{c1})->({r2},{c2}): 残差 = {v:+.12f} "
              f"({v / TWO_PI:+.3f}·2π)")
    # 精确关系：涡核环四条边残差（取逆时针方向）之和与包裹环和大小相等、
    # 符号相反（残差定义方向导致），绝对值恰为 2π——矛盾可定位、不可平滑消除。
    ccw = (
        before[(0, 0, 0, 1)]
        + before[(0, 1, 1, 1)]
        - before[(1, 0, 1, 1)]
        - before[(0, 0, 1, 0)]
    )
    print(f"  涡核环四边逆时针残差之和 = {ccw:+.12f}，"
          f"与包裹环和 {loop_sums[(0, 0)]:+.12f} 大小相等、符号相反")

    res = unwrap_phase(vortex, mask)
    print("切口后:")
    print(f"  残差              : {[(x.row, x.col, x.charge) for x in res.residues]}")
    print(f"  切口边数          : {len(res.cuts)}（从涡核环到最近外边界）")
    vr, vc = res.residues[0].row, res.residues[0].col
    touched = [e for e in res.cuts if cut_touches_plaquette(e, vr, vc)]
    print(f"  与涡核环相邻的切口: {len(touched)} 条")
    print(f"  连通区数          : {len(res.regions)}")
    print(f"  未满足边残差数    : {len(res.unmet_edges)}（非切口边全部一致）")
    worst_after = max((abs(e.residual) for e in res.unmet_edges), default=0.0)
    print(f"  非切口边最大残差  : {worst_after:.3e} rad（口径 1e-9）")
    # 朴素洪泛的矛盾边依赖其 DFS 生成树，与算法选出的“到外边界最短切口”
    # 不必是同一条边；共同事实是：切口前必有边背着 ±2π 残差，切口后
    # 所有非切口边残差归零，矛盾由切口边显式承担（不再参与一致性检查）。
    print("  实际切口边        :", [str(e) for e in res.cuts])
    print("  结论: ±2π 的矛盾被一条到外边界的切口显式承担，"
          "而不是被平滑结果掩盖。")

    # --- 3a. 实际触发：NaN 输入被带坐标拒绝 -------------------------------
    section("3a. 异常边界：NaN 输入必须被拒绝（可定位、不静默纠正）")
    bad_phase = [[0.1 * c for c in range(5)] for _ in range(5)]
    bad_phase[3][2] = float("nan")
    try:
        unwrap_phase(bad_phase, [[True] * 5 for _ in range(5)])
    except InvalidPhaseInputError as exc:
        print(f"  实际捕获 {type(exc).__name__}: {exc}")
        print(f"  定位 row={exc.row}, col={exc.col}")

    # --- 3b. 实际触发：孤岛涡旋 = 证明无解 --------------------------------
    section("3b. 异常边界：无效海孤岛内的单个涡旋 —— 证明无解")
    h = w = 11
    island = [[False] * w for _ in range(h)]
    for r in range(4, 7):
        for c in range(4, 7):
            island[r][c] = True
    # 岛内造一个 +1 残差（直接放置一个解析涡核在环 (4,4) 角点附近）。
    g = [[0.0] * w for _ in range(h)]
    for r in range(4, 7):
        for c in range(4, 7):
            g[r][c] = wrap_phase(
                math.atan2(r - 4.5, c - 4.5)
            )
    # 仅环 (4,4) 四点全有效时才是孤立残差；上面 3x3 岛的角环即 (4,4)。
    try:
        unwrap_phase(g, island)
    except NoSolutionError as exc:
        print(f"  实际捕获 {type(exc).__name__}")
        print(f"  信息: {exc}")

    # --- 3c. 实际触发：预算耗尽 ≠ 无解 ------------------------------------
    section("3c. 异常边界：搜索预算耗尽（与“证明无解”严格区分）")
    g2 = [
        [wrap_phase(math.atan2(r - 4.5, c - 4.5)) for c in range(9)]
        for r in range(9)
    ]
    mask2 = [[True] * 9 for _ in range(9)]
    try:
        unwrap_phase(g2, mask2, search_budget=1)
    except SearchBudgetExhausted as exc:
        print(f"  budget=1 时实际捕获 {type(exc).__name__}: {exc}")
    ok = unwrap_phase(g2, mask2, search_budget=1_000_000)
    print(f"  同一输入预算充足时正常完成：残差 {len(ok.residues)} 个，"
          f"未满足边 {len(ok.unmet_edges)} 条 —— 耗尽是“没算完”，不是无解。")

    # --- 4. 相邻两帧全局整数周对齐 align_frames ----------------------------
    section("4a. 相邻两帧对齐：跨 2π 时间跳变 + 局部缺测（只用共同像素）")
    # 3x3 网格上只有对角像素有效；A 帧取值 +pi-0.01，B 帧同一物理点取到
    # -pi+0.01（包裹在 2π 边界两侧，物理上相差一个整周）。中心 (1,1) 在
    # B 帧缺测（局部缺测），另带一个 B 独占像素，二者都不得计入 SSE。
    eps = 0.01
    ga = [[PI - eps, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, PI - eps]]
    gb = [[-PI + eps, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -PI + eps]]
    ma = [[True, False, False], [False, True, False], [False, False, True]]
    mb = [[True, False, False], [False, False, True], [False, False, True]]
    al = align_frames(ga, ma, gb, mb)
    print(f"  A 有效像素          : {[(r, c) for r in range(3) for c in range(3) if ma[r][c]]}")
    print(f"  B 有效像素          : {[(r, c) for r in range(3) for c in range(3) if mb[r][c]]}")
    print(f"  共同有效像素        : {list(al.shared_pixels)}（中心 (1,1) B 缺测，被排除）")
    print(f"  全局整数周偏移 k    : {al.shift}")
    print(f"  对齐后 SSE          : {al.squared_error:.6f} rad²")
    # 手算核对：共同像素 d=2pi-0.02；k=1 残差 -0.02，SSE=2*(0.02)^2=8e-4；
    # k=0 的 SSE 约 2*(2pi)^2≈78.96，故 k=1 唯一最优。
    d0 = ga[0][0] - gb[0][0]
    sse_k0 = 2.0 * d0 * d0
    print(f"  手算核对: d={d0:+.4f}，SSE(k=0)={sse_k0:.4f}，"
          f"SSE(k=1)=2*(0.02)^2={2 * 0.02 ** 2:.4f} -> k=1 唯一最优")
    for r, c in al.shared_pixels:
        print(f"  共同像素 ({r},{c}): phi_a={al.frame_a.unwrapped[r][c]:+.4f}, "
              f"phi_b+2pi*k={al.frame_b.unwrapped[r][c] + al.shift * TWO_PI:+.4f}")

    section("4b. 并列取较小整数：两个共同像素差恰为 +pi（mean/2pi=0.5）")
    ga2 = [[0.0, 0.0]]
    gb2 = [[-PI, -PI]]
    m2 = [[True, True]]
    al2 = align_frames(ga2, m2, gb2, m2)
    print(f"  SSE(k=0)=SSE(k=1)=2*pi^2={2 * PI * PI:.6f}；实际选择 k={al2.shift}"
          f"（并列取较小整数），SSE={al2.squared_error:.6f}")

    section("4c. 无共同有效像素：未对齐状态（不是异常），掩码保留")
    ga3 = [[0.1, 0.2], [0.3, 0.4]]
    gb3 = [[0.5, -0.5], [1.0, -1.0]]
    ma3 = [[True, False], [False, True]]
    mb3 = [[False, True], [True, False]]
    al3 = align_frames(ga3, ma3, gb3, mb3)
    print(f"  aligned={al3.aligned}, shift={al3.shift}, "
          f"squared_error={al3.squared_error}")
    print(f"  A 展开网格（无效位 None）: {al3.frame_a.unwrapped}")
    print(f"  B 展开网格（无效位 None）: {al3.frame_b.unwrapped}")

    section("4d. 实际错误：两帧形状不一致 —— 明确拒绝，无半更新状态")
    try:
        align_frames(
            [[0.1, 0.2], [0.3, 0.4]], [[True, True], [True, True]],
            [[0.1, 0.2, 0.3]], [[True, True, True]],
        )
    except InvalidPhaseInputError as exc:
        print(f"  实际捕获 {type(exc).__name__}: {exc}")
        print("  两帧输入均未被修改（纯函数），也不存在半完成的对齐结果。")

    elapsed = time.perf_counter() - t0
    print("\n" + "-" * 68)
    print(f"演示完成，实际计算耗时 {elapsed:.3f} 秒。")


if __name__ == "__main__":
    main()
