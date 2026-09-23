# phasecut — 图像相位展开的残差切口（residual branch-cut）模块

离线、确定性、仅用 Python 标准库实现的二维包裹相位残差切口展开。面向小尺寸
（≤ 32×32）带有效像素掩膜的包裹相位图，不调用任何第三方同类算法。

- **Python**：3.14.7（仅使用 3.10+ 通用语法；开发与验证均在 3.14.7）
- **依赖**：无。仅标准库（`math`、`dataclasses`、`typing`）。装好 Python 即可运行，
  不需要网络、账号、密钥、硬件、数据库、中间件、云 API 或 Docker。
- **测试/演示**：全部离线；示例数据由仓库内确定性代码生成。

本模块只实现本题契约所定义的数学子集（规则网格、4-邻域、单位曼哈顿对偶路径、
2×2 环路整数残差），**不宣称**与 Goldstein 分支切割等完整行业算法兼容。

## 运行命令

```bash
python -m unittest discover -s tests -v   # 必要测试（独立参考，见 tests/_reference.py）
python demo.py                            # 固定输入演示，约 0.05 秒（远低于 8 秒）
```

## 接口

```python
from phasecut import unwrap, residue_map, wrapped_difference

result = unwrap(phase, mask=None, *, max_expansions=10_000_000,
                on_infeasible="report")
```

- `phase`：`H` 行 `W` 列的 `list[list[float|int]]`，
  `1 <= H, W <= 32`，每个值必须有限且落在半开区间 `[-pi, pi)`。
- `mask`：可选，同形状；元素为 `bool` 或整数 `0/1`。`True` 表示有效测量像素。
  缺省为全有效。
- `max_expansions`：所有 BFS 扩展次数之和的硬上限（正整数，`bool` 拒绝）。
- `on_infeasible`：`"report"`（默认，无法处理的电荷进入 `unsatisfied`）或
  `"raise"`（抛 `UnwrapInfeasibleError`）。
- 输入不会被修改；任何失败都整体返回异常，不产生部分结果对象。

`residue_map(phase, mask=None)` 单独返回每个 2×2 面的整数残差网格
（无效面/无电荷面记为 0）。`wrapped_difference(d)` 把任意差绕回 `[-pi, pi)`。

## 返回数据结构（`UnwrapResult`，不可变 dataclass）

| 字段 | 含义 |
| --- | --- |
| `rows`, `cols` | 网格尺寸 |
| `wrapped`, `mask` | 归一化后的输入副本（float/bool 元组） |
| `residues` | 每个非零残差：`Residue(row, col, charge, wrapped_sum)` |
| `residue_grid` | 整数残差网格，无效面为 0 |
| `dipoles` | 异号配对：`Dipole(positive, negative, path, edges)`；`path` 是对偶面序列，`edges` 是路径跨过的像素边（长度 = `len(path)-1`） |
| `boundary_leashes` | 剩余电荷连外边界：`BoundaryLeash(source, charge, path, edges, exit)` |
| `cuts` | 全部不可跨切口像素边（`Edge`，含出口边），字典序排序 |
| `components` | 每个切口连通区：`Component(seed, cells, offsets, unwrapped)`；`seed` 为字典序最小像素，其偏移基准为 0，`cells` 字典序排序并与 `offsets`/`unwrapped` 对齐 |
| `offsets_grid` / `unwrapped_grid` | 每像素的 2π 整数层偏移与展开值；无效像素为 `None` |
| `unsatisfied` | 周边没有任何切口边的带电面（见下） |
| `edge_residuals` | **未满足边残差**：所有“存在且非切口”但积分残差非零的边 |

`Edge(a, b)` 的端点自动按字典序归一化（无向）。

## 算法与设计取舍

1. **残差**。对四角像素 `a=(r,c), b=(r,c+1), e=(r+1,c+1), d=(r+1,c)` 全在
   mask 内的 2×2 面，按顺时针累加四条边的包裹邻差
   `wrap(b-a)+wrap(e-b)+wrap(d-e)+wrap(a-d)`，取最接近 `S/2π` 的整数为电荷 `q`。
   四角任一无效则该面无效（残差 0，且面不参与路由）。
2. **对偶图路由**。面（2×2 环路）是对偶顶点；在两个面之间走一步跨过一条像素边，
   被跨过的边即切口。这样首/末条切口边必然落在被放电荷环路自身的周界上，真正
   放电。剩余电荷经边界面跨过一条网格边界像素边“连到外边界”。
3. **配对与字典序**。按残差位置字典序处理：每个电荷在“不重复使用切口边、不穿过
   其它带电面”的约束下，用分层 BFS 找最短曼哈顿对偶路径连到异号电荷；等距时取
   字典序最小的目标面，再取字典序最小的路径（固定 上→左→右→下 扩展序、逐层首占
   保证）。配不上的电荷再以同样规则连外边界。
4. **洪泛展开**。在被切口分割后的有效像素图上，每个连通区从字典序最小像素开始
   按 上→左→右→下 固定顺序洪泛；种子偏移基准为 0，其余像素沿包裹邻差积分，
   `unwrapped = wrapped + offset`，`offset` 恒为 `2π` 的整数倍。
5. **不掩盖矛盾**。若一个带电面周界上没有任何切口（例如它被无效像素围死，既配
   不到异号也到不了边界），它进入 `unsatisfied`，并给出四条周边边各自的残差；
   此外对每一条“存在且非切口”的边独立扫描，凡积分差与包裹邻差之差非零都列入
   `edge_residuals`（典型是绕数被 mask 洞遮蔽，没有任何有效 2×2 面能承载电荷的
   情形）。平滑外观不会被当成正确性证据。

### 错误语义（可定位，不静默纠正）

- `InputValidationError(ValueError)`：NaN/Infinity、`bool` 相位值、非数值、越界
  `[-pi,pi)`、尺寸越界/形状不一致、空行、mask 元素非 bool/0-1、`max_expansions`
  非正整数或为 `bool`、`on_infeasible` 非枚举值。错误信息带定位，如
  `phase[2][1]: value must be finite, got nan`（`exc.loc == "phase[2][1]"`）。
- `SearchBudgetExhaustedError`：BFS 扩展预算耗尽。**仅表示未能判定**，不断言
  有解或无解；与下面的“证明无解”严格区分。
- `UnwrapInfeasibleError`：穷尽搜索证明某电荷既到不了异号电荷也到不了外边界
  （`on_infeasible="raise"` 时）；`exc.unreachable` 给出对应面坐标。
- 默认 `"report"` 不抛无解异常，而是把矛盾放进 `unsatisfied` / `edge_residuals`。

## 浮点与误差口径

- 包裹运算为 `(x + π) mod 2π − π`（半开 `[-π, π)`）。
- 残差是**整数判定**：以 `floor(S/2π + 0.5)` 取整。合法包裹数据中单位绕数对应
  `|S ∓ 2π|` 在 32×32 规模上为双精度舍入量级（约 1e-15），远离 ±0.5 的取整边界。
  本模块不做亚像素噪声建模，也不提供容差调节来“放过”边界错误。
- 展开正确性的误差口径：
  - 无涡旋斜坡：全部像素 `|unwrapped − 真值|` 实测 ≤ 2e-15 rad；
  - 任一**未被切口接触**的 2×2 面：四条边（积分差 − 包裹邻差）之和绝对值
    ≤ `CONTRADICTION_TOL = 1e-9` rad；
  - 偏移量与 2π 整数层的偏差 ≤ 1e-11（路径 ≤ 32 步，见测试）；
  - 切口边两侧的积分差是精确的非零整数倍 2π（测试断言
    `|residual − k·2π| ≤ 1e-9, k ≠ 0`）。
- 阈值 `1e-9` 比双精度累积误差（< ~1e-13）大约四个数量级，比一次分支跳变
  （≈6.28）小约十个数量级；它只用于区分“数值零”和“真实分支矛盾”，不用宽
  容差掩盖边界错误。

## 测试如何得到期望值（独立参考约束）

`tests/_reference.py` 不导入被测核心：它独立实现包裹、独立环路残差公式、解析斜坡
（已知真值）与解析绕数场 `atan2`（已知整数绕数），以及无切口的朴素洪泛。测试的
期望电荷/真值来自这些构造与独立公式；对被测结果只做结构不变量核对（层偏移、
切口拓扑、电荷记账、未满足边残差），往返一致仅作补充。随机用例使用固定种子。

覆盖：无涡旋斜坡精确还原；单涡旋边界连边与切口前后环路误差；异号直连配对；
等距字典序选择；已有切口阻挡后的绕行；mask 孤岛与独立连通区基准；2×2 极小涡旋
贴近 ±π 边界；强梯度跨 π 场；mask 洞遮蔽绕数 → 未满足边残差；NaN/Infinity/bool/
越界等拒绝；预算耗尽与证明无解分离；32×32 边界接受；确定性重复调用。

## 已知限制

- 仅规则矩形网格、4-邻域、最近整数残差；不支持加权/质量图、多频/时间维展开、
  最小二乘或网络流全局最优。配对是题面指定的字典序贪心最短路径，不保证全局
  切口总长最小，也不实现 Goldstein 的邻域扩张平衡策略。
- 单个合法 2×2 环路的残差数学上只能是 −1、0、+1（四条包裹邻差之和无法达到
  ±4π 且绕环必须闭合），因此模块只处理单位电荷；不实现高阶电荷展开。
- 被 mask 洞遮蔽、无任何有效 2×2 面承载的绕数无法被“放电”，只会如实在
  `edge_residuals` 中报告——这是契约要求的显式矛盾，而不是被修平的结果。
- 连外边界的“出口边”是位于网格边界上的一条真实像素边（两端都是真实像素），
  不使用网格外虚拟坐标；洪泛只在真实像素间进行。

## 目录

```
phasecut/__init__.py     # 公共 API
phasecut/core.py         # 校验、残差、对偶路由、切口、洪泛、矛盾报告
tests/_reference.py      # 独立参考（不导入被测核心）
tests/test_validation.py # 输入与错误语义
tests/test_algorithm.py  # 斜坡/涡旋/配对/mask/跨π
tests/test_invariants.py # 结构不变量与确定性随机核对
demo.py                  # 固定输入演示
```
