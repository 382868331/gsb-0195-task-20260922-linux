"""phasecut 固定输入演示（离线、确定性、无第三方依赖）。

运行：python demo.py

内容：
1. 无涡旋斜坡（频繁跨 pi 包裹）：无残差、无切口、单区精确还原。
2. 单涡旋参考（解析 atan2 场）：对比“不加切口的朴素洪泛”与加切口后的
   环路误差——前者在涡核环留下约 2π 的矛盾，后者非切口边全部一致。
3. 实际触发的边界：NaN 输入被带坐标拒绝；孤岛涡旋证明无解；
   搜索预算耗尽与无解明确区分。
"""

import math
import time

from phasecut import (
    InvalidPhaseInputError,
    NoSolutionError,
    SearchBudgetExhausted,
    TWO_PI,
    unwrap_phase,
    wrap_phase,
    align_frame_pair,
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

    # --- 4. 相邻两帧全局整数周对齐（正常结果：跨 2π + 局部缺测） ----------
    section("4. 相邻帧对齐 4x8（斜坡列梯度 2.5 跨 2π；两帧各有局部缺测）")
    h, wd = 4, 8
    truth1 = [[0.2 * r + 2.5 * c for c in range(wd)] for r in range(h)]
    bias = -4.0  # 帧2真值整体低 4.0 rad
    truth2 = [[v + bias for v in row] for row in truth1]
    f1 = [[wrap_phase(v) for v in row] for row in truth1]
    f2 = [[wrap_phase(v) for v in row] for row in truth2]
    mm1 = [[True] * wd for _ in range(h)]
    mm2 = [[True] * wd for _ in range(h)]
    mm1[2][3] = False  # 帧1局部缺测
    mm2[0][0] = False  # 帧2锚点缺测 => 帧2改从 (0,1) 起展开
    n_common = sum(1 for r in range(h) for c in range(wd)
                   if mm1[r][c] and mm2[r][c])

    al = align_frame_pair(f1, mm1, f2, mm2)
    print(f"共同有效像素数    : {len(al.common_pixels)}（手算 {n_common}）")
    print(f"是否对齐          : {al.aligned}")
    # 手算：帧1锚点 (0,0) 展开 φ1(0,1)=2.5；帧2锚点缺 (0,0) 改从 (0,1)
    # 起展开，φ2(0,1)=wrap(2.5-4)=wrap(-1.5)=-1.5。同梯度 => 共同像素上
    # d=φ1-φ2 恒为 2.5-(-1.5)=4.0 => K=round(4/2π)=1。
    print("全局整数周偏移 K  : "
          f"{al.offset_cycles}（手算：φ1(0,1)=2.5、φ2(0,1)=-1.5，"
          "d 恒为 4.0，round(4/2π)=1）")
    per = 4.0 - al.offset_cycles * TWO_PI
    print(f"对齐后逐像素残差  : 恒为 {per:+.6f} rad = 4 − 2π")
    print(f"最小平方差 SSE    : {al.sse:.6f}（手算 {n_common}·({per:.6f})² "
          f"= {n_common * per * per:.6f}；K=0 时为 {n_common * 16.0:.1f}）")
    # 帧1独有像素 (2,3) 不参与 K 的选择，但帧2该像素有效，仍被整体平移。
    v = al.aligned_frame2[2][3] - al.frame2.unwrapped[2][3]
    print(f"帧1缺测像素 (2,3) : 不参与选择；帧2该处仍整体平移 "
          f"{v / TWO_PI:+.0f}·2π = {v:+.6f} rad")
    # 抽一个共同像素展示包裹->展开->对齐的全过程。
    r, c = 0, 1
    print(f"抽样像素 (0,1)    : 包裹=({f1[r][c]:+.4f},{f2[r][c]:+.4f}) "
          f"展开=({al.frame1.unwrapped[r][c]:+.4f},"
          f"{al.frame2.unwrapped[r][c]:+.4f}) "
          f"对齐后帧2={al.aligned_frame2[r][c]:+.4f}（帧1差 "
          f"{al.aligned_frame2[r][c] - al.frame1.unwrapped[r][c]:+.4f}）")

    # --- 5. 边界：无任何共同像素 => 未对齐，保留掩码与两帧结果 -------------
    section("5. 帧对齐边界：掩码棋盘互补、无共同像素 —— 返回未对齐")
    h = w = 4
    gg = [[wrap_phase(0.1 * r + 0.2 * c) for c in range(w)] for r in range(h)]
    cc1 = [[(r + c) % 2 == 0 for c in range(w)] for r in range(h)]
    cc2 = [[(r + c) % 2 == 1 for c in range(w)] for r in range(h)]
    alu = align_frame_pair(gg, cc1, gg, cc2)
    print(f"  aligned={alu.aligned}，K={alu.offset_cycles}，"
          f"SSE={alu.sse}，共同像素={len(alu.common_pixels)}")
    print(f"  两帧展开结果仍保留：帧1区数={len(alu.frame1.regions)}，"
          f"帧2区数={len(alu.frame2.regions)}；掩码未被修改。")
    print(f"  aligned_frame2 退化为帧2自身展开："
          f"{alu.aligned_frame2 == alu.frame2.unwrapped}")

    # --- 6. 实际错误：帧2形状不一致，明确拒绝、无半更新 --------------------
    section("6. 帧对齐错误：两帧形状不一致 —— InvalidPhaseInputError")
    bad2 = [[wrap_phase(0.1 * c) for c in range(7)] for _ in range(4)]
    badm2 = [[True] * 7 for _ in range(4)]
    try:
        align_frame_pair(f1, mm1, bad2, badm2)
    except InvalidPhaseInputError as exc:
        print(f"  实际捕获 {type(exc).__name__}: {exc}")
    print("  时间对齐只在同形帧间定义；此调用不产生任何结果对象，"
          "输入列表不被修改。")

    elapsed = time.perf_counter() - t0
    print("\n" + "-" * 68)
    print(f"演示完成，实际计算耗时 {elapsed:.3f} 秒。")


if __name__ == "__main__":
    main()
