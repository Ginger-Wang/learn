"""``sources/skirtBellCollider.h`` / ``skirtBellCollider.cpp`` 的 Python 移植版
（Maya Python API 1.0）。

由髋/膝/踝关节驱动的一叠钟形碰撞器，放样成单一的周期性 NURBS 曲面——即裙子。

与 C++ 版的关键差异：**不做视口绘制**。C++ 版用 ``MPxDrawOverride`` 把碰撞环
画在视口里；这里改成输出真实网格：

* ``outputSurface`` —— 裙子的 NURBS 曲面（C++ 版就有这个属性）
* ``outputRingMesh`` —— 全部碰撞环合并成的一个网格（**新增**）

整体思路（读代码前先看这一段，后面每个函数只讲自己那一步）
--------------------------------------------------------

这是本工程里最复杂的节点：``bellCollider`` 只解**一个**钟形，本节点要解
**一叠**钟形，再把它们逐层的轮廓当作 NURBS 曲面的控制点行，放样成一条裙子。

一次 :meth:`SkirtBellCollider.compute` 的流水线：

1. **量身**（:func:`computeSkirtLayout`）—— 从髋/膝/踝六个关节矩阵量出大腿、
   小腿的**刚性**长度，换算出"腰 → 髋 → 膝 → 踝"沿裙子轴的一串距离
   ``levelDistances``。短裙 2 层、长裙 3 层。
2. **造碰撞环**（:func:`_createRingMatrix`）—— 每条腿由髋关节出发、指向膝/踝
   造一个胶囊状的环矩阵。注意环是**从髋部长出来的一根长柱**（缩放 Y 取整段
   骨长），不是在膝盖上放一个小圈 —— 整条腿都要能顶到裙子。
3. **逐层求解**（:class:`~bellColliderSolver.BellColliderSolver`）—— 每层用
   本层的 ``bellMatrix`` + 上面那几个环调一次 ``solve``，从返回网格里取出
   顶圈（第 0 层还要取底圈）当作该层的曲面控制点行。
4. **放样成面** —— U 方向沿一圈（3 次周期），V 方向沿高度（1 次开放），
   V 方向的 knot 直接用真实层距离，这样参数化和实际尺度一致。

贯穿全文的**矩阵语义约定**（与 ``bellColliderSolver`` 一致）：矩阵的第 1 行
（Y 轴）既是朝向、其长度又是高度（钟形）或半径（环），X/Z 行的长度是横向
缩放，第 3 行是位置。所以下面到处出现的 ``X *= scale.x`` 之类不是"设置缩放
属性"，而是**直接把尺寸编码进轴长**。
"""

import maya.OpenMaya as om
import maya.OpenMayaMPx as ompx

from utils import (maxis, taxis, xaxis, zaxis, lerp_point, midpoint,
                   matrix_from_rows, ramp_value_at, mesh_points,
                   create_numeric3, input_matrix, plug_vector)
from bellColliderSolver import (BellColliderInputs, BellColliderOutputs,
                                BellColliderSolver)


NODE_NAME = "skirtBellCollider"

# 每层钟形"底圈半径 / 顶圈半径"的上限。见 compute 里用到的地方：
# 这个比值来自 bellScaleRamp，顶部接近 0 时会趋于无穷。
_MAX_BOTTOM_RATIO = 100.0

# 打开后每次 compute 都会把读到的关键参数打到 Script Editor。
# 排查"属性值和实际效果对不上"时最有用 —— 直接看 compute 到底读进去了什么，
# 不用从外面猜。查完记得关掉：它每次求值都打印，会很吵。
#
#     import skirtBellCollider
#     skirtBellCollider.DEBUG = True
DEBUG = False


def _getCurvePoints(points, bellSubdivision, use_bottom):
    """从一个已解算的钟形网格里抠出一圈顶点，做成**曲面某一行**的控制点。

    ``compute`` 里每层调用一次（第 0 层调两次：底圈 + 顶圈），返回值直接成为
    ``rows[v]``，也就是最终 NURBS 曲面在 V 方向第 v 行的 CV。

    下标约定沿用求解器的钟形顶点布局：0 是底面中心点，``1..bellSubdivision``
    是底圈，其后是顶圈。所以 `use_bottom` 只是在切换起始下标 ——
    ``1``（底圈）还是 ``bellSubdivision + 1``（顶圈）。只有最底下那层需要底圈，
    因为相邻两层是首尾相接的：上一层的顶圈就是下一层的底圈，取两次会重复。

    **为什么要多出 3 个 CV**：曲面 U 方向是 3 次（cubic）``kPeriodic``。
    Maya 的周期形式要求把开头 ``degree`` 个 CV 原样重复到末尾来闭合，
    3 次就是重复 3 个 —— 所以数组长度是 ``bellSubdivision + 3``，
    ``compute`` 里的 ``numU`` 正是这个数。少一个都会让 create 失败或接缝处
    出现折角。

    这里用 ``setLength`` + ``set(v, i)`` 而不是 ``append``：API 1.0 的
    MPointArray 没有 ``arr[i] = v``，写元素必须用 ``set(值, 下标)``（参数顺序
    是值在前）。
    """
    start = 1 if use_bottom else bellSubdivision + 1
    count = bellSubdivision

    cvs = om.MPointArray()
    cvs.setLength(count + 3)
    for i in range(count):
        cvs.set(points[start + i], i)
    # 重复前 3 个 CV，保证三次周期曲线的连续性
    cvs.set(cvs[0], count)
    cvs.set(cvs[1], count + 1)
    cvs.set(cvs[2], count + 2)
    return cvs


def _clearRampEntries(rampAttr):
    """删掉渐变曲线上现有的全部条目。

    只能在**节点已经建好之后**调用 —— ``postConstructor`` 那个时机 ramp 的
    数据还不存在，这里的调用会直接抛 "Object does not exist"。

    只被 :func:`resetBellScaleRamp` 调用。``deleteEntries`` 要的是**条目索引**
    数组，而索引只能靠 ``getEntries`` 拿到，所以哪怕只想删东西也得先把
    position / value / interpolation 三个数组一起读出来（API 1.0 的
    out-param 风格：四个数组都是传进去被填的）。
    """
    if rampAttr.getNumEntries() == 0:
        return

    indices = om.MIntArray()
    positions = om.MFloatArray()
    values = om.MFloatArray()
    interpolations = om.MIntArray()
    rampAttr.getEntries(indices, positions, values, interpolations)

    if indices.length() > 0:
        rampAttr.deleteEntries(indices)


def resetBellScaleRamp(nodeName, positions=(0.0, 1.0), values=(1.0, 1.0)):
    """把某个 skirtBellCollider 的 ``bellScaleRamp`` 重置成干净的曲线。

    **建完节点后需要调一次**，原因是 ramp 的默认条目不受插件控制：

    * ``postConstructor`` 里访问 ramp 会报 "Object does not exist" ——
      那时属性数据还没建好，所以没法在那里清理。
    * Maya 会在节点创建之后自己补一条默认斜坡 ``(0, 0) -> (1, 1)``，
      它和 ``postConstructor`` 用 ``addEntries`` 加进去的条目**并存**。

    结果 position 0 处同时有 ``(0, 0)`` 和 ``(0, 1)`` 两个冲突条目，求值可能
    返回 0。而 0 会让 compute 里 ``scale_bottom / scale_top`` 这个比值冲到
    1e5 量级，裙子半径直接变成天文数字（表现为曲面炸成一片延伸到天边的条纹）。

    默认参数给的是一条恒为 1 的水平线，也就是不额外缩放半径 —— 与 C++ 版
    ``postConstructor`` 的意图一致。

        import skirtBellCollider
        skirtBellCollider.resetBellScaleRamp("skirtBellCollider1")

    这是给用户/脚本手动调的工具函数，**节点自身从不调用它**（节点无从知道
    Maya 什么时候补完默认条目）。
    """
    # 按名字找到节点：API 1.0 没有 "by name 拿 MObject" 的直接接口，
    # 标准做法就是过一遍 MSelectionList（getDependNode 也是 out-param）。
    sel = om.MSelectionList()
    sel.add(nodeName)
    obj = om.MObject()
    sel.getDependNode(0, obj)

    # 先清空，再整条重建 —— 不去逐条改，因为冲突条目的索引不可预测
    rampAttr = om.MRampAttribute(obj, SkirtBellCollider.attr_bellScaleRamp)
    _clearRampEntries(rampAttr)

    # 三个数组必须等长且一一对应：位置、值、该点的插值方式
    posArray = om.MFloatArray()
    valArray = om.MFloatArray()
    interpArray = om.MIntArray()
    for p, v in zip(positions, values):
        posArray.append(float(p))
        valArray.append(float(v))
        interpArray.append(om.MRampAttribute.kLinear)

    rampAttr.addEntries(posArray, valArray, interpArray)

    print("%s.bellScaleRamp 已重置为 %d 个条目：%s"
          % (nodeName, len(positions),
             ", ".join("(%.3f, %.3f)" % (p, v)
                       for p, v in zip(positions, values))))


def _getAxis(m, idx):
    """按枚举下标从矩阵里取一根轴，支持负方向。

    ``leftRingAxis`` / ``rightRingAxis`` / ``bellAxis`` 三个枚举属性的取值
    直接就是这里的 `idx`，六个选项对应"哪根轴 + 正还是负"。之所以需要负向，
    是因为骨骼轴的指向完全取决于绑定习惯：腰部关节的 Y 通常朝上，而裙子要
    朝下长，所以 ``bellAxis`` 常选 -Y；左右腿互为镜像，所以默认
    ``leftRingAxis=0``（X）而 ``rightRingAxis=3``（-X）。

    **返回的向量保留原始长度**（``maxis`` 不做归一化），调用方自己决定要不要
    ``normal()`` —— 因为轴长在本工程里是有意义的（缩放/高度），不能随手丢掉。
    """
    # 0=X, 1=Y, 2=Z, 3=-X, 4=-Y, 5=-Z
    v = maxis(m, idx % 3)
    return -v if idx >= 3 else v


class SkirtLayout(object):
    """一次求解需要的全部"层"信息：起点、方向、各层沿轴距离。

    从 compute 里抽出来是为了能被诊断脚本原样复用 —— 诊断和实际计算走同一份
    代码，才不会出现"诊断说没问题、实际却是错的"。

    这是个**纯数据容器**，只由 :func:`computeSkirtLayout` 负责填充，本身不带
    任何方法。用 ``__slots__`` 而不是普通字典：字段是固定的一组，写错名字要
    立刻 AttributeError，不能悄悄多出一个拼错的属性（这类错误在几何代码里
    表现为"某个量恒为默认值"，极难发现）。

    字段含义（长度类字段的单位都是场景单位，沿 `dir_vector` 量）：

    * `W` —— 腰（waist）的位置，取自 ``bellMatrix`` 的平移，是裙子的**起点**。
    * `H` —— 左右髋关节的中点。
    * `L_thigh` / `L_calf` —— 大腿、小腿的长度，取左右腿的平均值。
    * `d_hip` / `d_knee` / `d_heel` —— 从腰算起到髋/膝/踝的**累计**距离。
    * `d_mid` —— 髋与膝的中点距离，也就是中间那层的位置。
    * `h_val` —— 裙摆（最后一层）落在哪儿，由 ``height`` 属性插值得到。
    * `s` —— 短裙用的整体缩放系数：``h_val / 默认全长``。
    * `dir_vector` —— 裙子往下长的**单位**方向。
    * `levelDistances` —— 从 0 开始、逐层递增的距离表，长度 ``N + 1``；
      第 i 层钟形的底在 ``levelDistances[i]``、顶在 ``levelDistances[i+1]``。
      它同时被拿去当 NURBS 曲面 V 方向的 knot 值。
    * `N` —— 钟形的层数：短裙 2，长裙 3。
    """

    __slots__ = ("W", "H", "L_thigh", "L_calf", "d_hip", "d_knee", "d_heel",
                 "d_mid", "h_val", "s", "dir_vector", "levelDistances", "N")


def computeSkirtLayout(inputBellMatrix, LH, LK, LHe, RH, RK, RHe,
                       skirtType, height, bellAxis):
    """算出裙子各层沿轴的距离，以及起点和方向。

    ``compute`` 流水线的第一步，也是唯一一处"读骨骼、定尺寸"的地方。
    参数 `LH`/`LK`/`LHe`/`RH`/`RK`/`RHe` 是左右髋、膝、踝的**位置点**
    （调用方已经用 ``taxis()`` 从各自的矩阵里取好了），`skirtType`
    0=短裙、1=长裙，`height` 是 0..1 的裙长比例，`bellAxis` 是枚举下标。

    两个关键的设计决定：

    1. **腿长用刚性骨骼长度累加，而不是"髋到踝的直线距离"。** 屈膝时后者会
       急剧变短，裙子会跟着一起缩上去；分段累加则和姿势无关，角色怎么动裙长
       都稳定。左右腿取平均是因为裙子只有一个，两条腿不一致时得折中。
    2. **只返回数据、不碰 dataBlock。** 所以诊断脚本可以直接喂矩阵进来复算，
       和 compute 得到完全一样的结果。

    返回填好的 :class:`SkirtLayout`。
    """
    layout = SkirtLayout()

    layout.H = midpoint(LH, RH)
    layout.W = taxis(inputBellMatrix)

    # 用刚性骨骼长度计算裙子高度，避免屈膝时被拉伸
    layout.L_thigh = ((LK - LH).length() + (RK - RH).length()) * 0.5
    layout.L_calf = ((LHe - LK).length() + (RHe - RK).length()) * 0.5

    # 注意：d_hip 是"腰到髋"的距离，量的是 bellMatrix 的位置到髋部中点。
    # bellMatrix 没连接时它是单位矩阵、W 落在世界原点，这个距离就变成
    # "角色髋部到原点"的距离 —— 裙子会从原点开始、沿轴拉出去很远，
    # 表现为曲面缩成一条极长的线。
    layout.d_hip = (layout.H - layout.W).length()
    layout.d_knee = layout.d_hip + layout.L_thigh
    layout.d_heel = layout.d_knee + layout.L_calf
    layout.d_mid = (layout.d_hip + layout.d_knee) * 0.5

    # height 属性本身有 0.01..1 的 UI 限制，但 UI 限制只管通道盒 —— 通过
    # 表达式或直接 setAttr 照样能写进越界值，所以这里再夹一次。
    h_param = float(height)
    if h_param < 0.0:
        h_param = 0.0
    if h_param > 1.0:
        h_param = 1.0

    # height 的含义随裙型而变：它是在"这一段区间"里插值，而不是全长的比例。
    # 所以长裙的 height=0 是及膝、=1 是及踝，短裙的 height=0 是齐髋、=1 是及膝。
    if skirtType == 1:  # 长裙：裙摆在膝盖到脚跟之间插值
        layout.h_val = layout.d_knee + (layout.d_heel - layout.d_knee) * h_param
    else:  # 短裙：裙摆在髋部到膝盖之间插值
        layout.h_val = layout.d_hip + (layout.d_knee - layout.d_hip) * h_param

    # s = 实际裙长 / 该裙型的"满长"，只有短裙用得上（见下面 levelDistances）。
    # 分母兜底成 1.0 是防除零：骨骼没连接时各距离可能全是 0。
    raw_defaultHeight = layout.d_heel if skirtType == 1 else layout.d_knee
    defaultHeight = 1.0 if raw_defaultHeight < 1e-5 else raw_defaultHeight
    layout.s = layout.h_val / defaultHeight

    # 根据选定的钟形矩阵轴取得方向向量。裙子是沿这个方向往下长的，
    # 所以对于 Y 轴指向骨骼末端（向上）的腰部关节，通常要选 -Y。
    raw_dir_vector = _getAxis(inputBellMatrix, bellAxis)
    layout.dir_vector = (om.MVector(0, -1, 0) if raw_dir_vector.length() < 1e-4
                         else raw_dir_vector.normal())

    # 层数：短裙 2 段（腰→中→膝），长裙 3 段（腰→中→膝→裙摆）。
    # 曲面 V 方向的 CV 行数是 N + 1。
    layout.N = 2 if skirtType == 0 else 3

    # levelDistances[0] 恒为 0：距离都是**从腰量起**的，起点就是腰。
    layout.levelDistances = [0.0]
    if skirtType == 1:  # 长裙：height 只控制脚跟那一层
        # 长裙的中间两层钉死在解剖位置（髋膝中点、膝盖），只有最后一层跟着
        # height 走。这样调裙长时上半截不会跟着一起伸缩，膝盖处的碰撞环始终
        # 对准膝盖。
        layout.levelDistances.append(layout.d_mid)
        layout.levelDistances.append(layout.d_knee)
        layout.levelDistances.append(layout.h_val)
    else:  # 短裙：height 缩放所有层
        # 短裙只有 2 层，如果也把中间层钉死，height 调小时两层会挤在一起、
        # 最后一段退化成零长度。所以整体按 s 等比缩放，层与层始终分得开。
        layout.levelDistances.append(layout.d_mid * layout.s)
        layout.levelDistances.append(layout.d_knee * layout.s)

    return layout


def _createRingMatrix(jointMatrix, scale, axisIndex=0, targetPos=None):
    """由一个关节造出喂给求解器的**碰撞环矩阵**。

    ``compute`` 里每条腿调 1~3 次，结果塞进 ``BellColliderInputs.ringMatrices``。
    求解器只认矩阵、不认骨骼，所以关节的姿态必须先在这里翻译成环的语义：

    * 第 3 行（平移）= 环心 = 关节位置。
    * 第 1 行（Y）= 环的**轴向**，长度 = 环的长度/高度。
    * 第 0/2 行（X/Z）的长度 = 环的**半径**（求解器建环时基础半径按 1）。

    参数：

    * `scale` —— 直接乘到三根轴上的尺寸。调用方传的是
      ``(ringScale.x, 骨长 * ringScale.y, ringScale.z)`` —— 也就是说
      **Y 分量承载的是整段骨头的长度**，环是从髋部一路伸到膝/踝的一根长柱，
      而不是套在关节上的一个小圈。整条腿都得能顶到裙子。
    * `axisIndex` —— ``leftRingAxis`` / ``rightRingAxis`` 枚举值，选关节的哪根
      轴当环的初始法线（见 :func:`_getAxis`）。
    * `targetPos` —— 可选的瞄准点（膝或踝）。给了就把 Y 转过去，让环真正沿着
      骨头方向躺下；不给则直接用关节自己的轴。

    做法是"取关节轴 → 朝目标旋转 → 重新正交化 → 乘缩放 → 拼矩阵"。**不直接
    用关节矩阵**，因为骨骼上常带非均匀缩放和剪切，那种矩阵拿去求逆做碰撞
    判定会算出错误的半径；这里重建出来的是一个干净的正交标架。

    三处退化保护（都会静默降级、不抛异常，见各自的行内注释）：反向 180 度、
    uX 与 Y 平行、以及零长度轴。
    """
    pos = taxis(jointMatrix)

    # 用选定的关节轴作为环的法线（即环矩阵中的 Y 行）
    rawY = _getAxis(jointMatrix, axisIndex)
    Y = rawY.normal()

    # 取出关节剩余的轴，以保留关节实际的旋转/扭转
    jointX = maxis(jointMatrix, (axisIndex + 1) % 3)
    uX = jointX.normal()

    # 若提供了 targetPos，把 Y 对齐到指向 targetPos 的方向
    if targetPos is not None:
        V = targetPos - pos
        if V.length() > 1e-6:
            Vn = V.normal()

            if Y * Vn < -0.9999:
                # Y 和目标方向**恰好反向**。这时不能用 rotateTo：180 度旋转的
                # 轴有无穷多个，实现只能任选一个，数值上很不稳定，会算出垃圾
                # 值甚至 NaN —— 环矩阵一坏，整条链路的坐标就爆到十万量级。
                #
                # 反向的情况根本不需要求旋转：把 Y 翻过来就已经是目标方向了。
                # uX 不用动，下面反正会重新正交化。
                #
                # 什么时候会撞上：ringAxis 选了沿骨骼方向的轴（比如关节 Y 轴
                # 指向下一个关节，却选了 -Y）。C++ 默认取 X / -X 正是为了避开
                # 这一点 —— 垂直于骨骼的轴转过去只有 90 度，稳定得多。
                Y = -Y
            else:
                Q = Y.rotateTo(Vn)
                rotMatrix = Q.asMatrix()
                Y = Y * rotMatrix
                uX = uX * rotMatrix

    # 把 uX 投影为与 Y 垂直
    rawX = uX - Y * (uX * Y)
    if rawX.length() < 1e-6:
        # uX 和 Y 平行，投影后剩零向量，X/Z 都会变成 0、环矩阵直接退化。
        # 另取一个确定垂直于 Y 的方向。
        seed = om.MVector(1.0, 0.0, 0.0)
        if abs(Y * seed) > 0.99:
            seed = om.MVector(0.0, 0.0, 1.0)
        rawX = seed ^ Y

    # 至此 X ⊥ Y，第三根轴用叉积补齐，得到一个干净的右手正交标架
    # （Y 此时还是单位向量，缩放在下面才乘上去）。
    X = rawX.normal()
    Z = (Y ^ X).normal()

    # 把尺寸编码进轴长 —— 这是本工程的矩阵语义：轴长即半径/长度。
    # 求解器（ringPush、deformPoints）就是靠"世界长度 / 局部长度"反推出
    # 这里写进去的缩放，从而知道环的实际半径。
    X *= scale.x
    Y *= scale.y
    Z *= scale.z

    # 第 4 列是 (0,0,0,1)：仿射矩阵、行向量约定（点写在矩阵左边）。
    # 必须走 matrix_from_rows —— API 1.0 不能用列表直接构造 MMatrix，
    # 而且 createMatrixFromList 在某些版本会静默失败（详见 utils）。
    return matrix_from_rows([[X.x, X.y, X.z, 0.0],
                             [Y.x, Y.y, Y.z, 0.0],
                             [Z.x, Z.y, Z.z, 0.0],
                             [pos.x, pos.y, pos.z, 1.0]])


class SkirtBellCollider(ompx.MPxLocatorNode):
    """``skirtBellCollider`` 节点：由腿部骨骼驱动、会躲开双腿的裙子曲面。

    继承 ``MPxLocatorNode`` 而不是 ``MPxNode``，是为了让节点在场景里有个可见、
    可选的定位器图标（C++ 版还靠它挂 ``MPxDrawOverride``；Python 版没有视口
    绘制，但保留同样的基类以便沿用同一套 UI 和场景文件）。

    **输入属性**

    * 七个矩阵：``bellMatrix``（腰）、``left/rightHipMatrix``、
      ``left/rightKneeMatrix``、``left/rightHeelMatrix``。腰决定裙子起点和朝向，
      其余六个用来量腿长、摆碰撞环。
    * ``skirtType`` —— 0=Short（2 层）/ 1=Long（3 层），默认 Long。
    * ``height`` —— 0.01..1，裙长比例，含义随裙型而变（见
      :func:`computeSkirtLayout`）。
    * ``ringScale`` / ``bellScale`` —— 3 分量缩放，前者调碰撞环的粗细，
      后者调裙子的胖瘦。**必须用 MPlug 读**（见 ``utils.plug_vector``）。
    * ``bellSubdivision`` / ``ringSubdivision`` —— 一圈的分段数。前者直接决定
      曲面 U 方向的 CV 数，后者只影响 ``outputRingMesh`` 的显示精度。
    * ``falloff`` / ``collision`` —— 透传给求解器，分别控制避让的角度范围和
      径向挤压强度。
    * ``tightness`` —— 本节点独有，只对长裙的最下面那层生效，见 ``compute``。
    * ``bellScaleRamp`` —— 沿裙子**高度**方向的半径渐变曲线。
    * ``leftRingAxis`` / ``rightRingAxis`` / ``bellAxis`` —— 六选一的轴枚举，
      用来适配不同的骨骼朝向约定（默认 X / -X / Y）。

    **输出属性**

    * ``outputSurface`` —— 裙子的 NURBS 曲面（U 三次周期、V 一次开放）。
    * ``outputRingMesh`` —— 全部碰撞环合并成的一个网格，纯显示用（新增）。

    **依赖关系**：``initialize`` 末尾把上面**每一个**输入都
    ``attributeAffects`` 到**两个**输出上。意思是任何一个输入变脏，两个输出
    都要重算 —— 没有做更细的粒度划分（比如 ``ringSubdivision`` 其实只影响
    ringMesh），因为 ``compute`` 本来就是一次算完两个输出的。

    **nodeId**：``om.MTypeId(0x01025)``。注意它**不等于** C++ 版的 1274436
    （0x01025 = 4133）—— 所以严格说两版的 skirtBellCollider 类型 ID 并不一致，
    但节点名相同，Maya 仍会拒绝同时注册两个同名节点（README 里"不要同时加载
    两个插件"那条依然成立）。

    类体里那一长串 ``attr_xxx = om.MObject()`` 是**类级**的属性句柄占位：
    Maya 要求属性对象在所有实例间共享，由 :func:`initialize` 在注册节点时
    填上真值。
    """

    typeId = om.MTypeId(0x01025)
    typeName = NODE_NAME

    # 输入属性
    attr_bellMatrix = om.MObject()
    attr_leftHipMatrix = om.MObject()
    attr_leftKneeMatrix = om.MObject()
    attr_leftHeelMatrix = om.MObject()
    attr_rightHipMatrix = om.MObject()
    attr_rightKneeMatrix = om.MObject()
    attr_rightHeelMatrix = om.MObject()

    attr_skirtType = om.MObject()
    attr_height = om.MObject()
    attr_ringScale = om.MObject()
    attr_bellScale = om.MObject()
    attr_bellSubdivision = om.MObject()
    attr_ringSubdivision = om.MObject()
    attr_falloff = om.MObject()
    attr_collision = om.MObject()
    attr_tightness = om.MObject()
    attr_bellScaleRamp = om.MObject()
    attr_leftRingAxis = om.MObject()
    attr_rightRingAxis = om.MObject()
    attr_bellAxis = om.MObject()

    # 输出属性
    attr_outputSurface = om.MObject()
    attr_outputRingMesh = om.MObject()

    def __init__(self):
        """节点实例的构造函数，由 :func:`creator` 调用。

        除了转调基类没有别的事可做：节点没有需要缓存的状态，所有数据都从
        dataBlock / MPlug 现读。**不能在这里碰属性** —— 这个时机
        ``thisMObject()`` 还没绑定到真正的 MObject 上。要在创建后做初始化，
        用 :meth:`postConstructor`。
        """
        ompx.MPxLocatorNode.__init__(self)

    def postConstructor(self):
        """节点对象建好之后 Maya 自动回调一次：给 ramp 铺上默认的两个端点。

        为什么必须在这里而不是 ``__init__``：只有到了 postConstructor，
        ``thisMObject()`` 才返回有效的 MObject，才能构造 ``MRampAttribute``。

        铺的是 (0, 1) 和 (1, 1) 两个点，即一条恒为 1 的水平线 —— 默认不对
        裙子半径做任何额外缩放。
        """
        thisObj = self.thisMObject()
        rampAttr = om.MRampAttribute(thisObj, SkirtBellCollider.attr_bellScaleRamp)

        # 注意：这里**不能**先清理已有条目。postConstructor 的时机 ramp 数据
        # 还没建好，getNumEntries / getEntries 会直接抛 "Object does not exist"。
        #
        # 而且 Maya 之后还会自己补一条默认斜坡 (0,0)->(1,1)，和这里加进去的
        # 条目并存，position 0 处就出现 (0,0) 和 (0,1) 两个冲突值 —— 求值可能
        # 返回 0，进而让 compute 里的半径比值爆炸。
        # 想要一条干净的曲线，请在建完节点后调用 resetBellScaleRamp()。
        positions = om.MFloatArray()
        values = om.MFloatArray()
        interpolations = om.MIntArray()

        positions.append(0.0)
        values.append(1.0)
        interpolations.append(om.MRampAttribute.kLinear)

        positions.append(1.0)
        values.append(1.0)
        interpolations.append(om.MRampAttribute.kLinear)

        rampAttr.addEntries(positions, values, interpolations)

    # -- 计算 ---------------------------------------------------------------

    def compute(self, plug, dataBlock):
        """DG 求值入口：把腰/腿的关节矩阵变成一张裙子曲面。

        Maya 在**任一输出插槽变脏、又被人要值**时调用（依赖关系见类
        docstring）。返回 ``om.kUnknownParameter`` 表示"这个插槽不归我管，
        走默认处理" —— 对 locator 节点必须这么写，否则 ``localScale`` 之类
        基类属性会失效。

        两个输出是**一次算完的**：``outputRingMesh`` 用的就是求解时那几个环
        矩阵，所以显示出来的环和实际参与碰撞的环必然一致，不会出现"看着没碰到
        却变形了"。末尾对两个输出都调 ``setClean``（而不是只 clean 传进来的
        `plug`），正是因为两个都已经算好了，否则另一个会被无谓地重算一遍。

        主要步骤（下面的行内注释逐段对应）：

        1. 读输入 —— 矩阵走 ``utils.input_matrix``（必须拷贝），3 分量属性走
           ``utils.plug_vector``（dataBlock 那条路读出来是垃圾值），其余标量
           照常走 dataBlock。
        2. 量身 —— :func:`computeSkirtLayout` 给出层数和各层距离。
        3. 造环 —— :func:`_createRingMatrix` 给每条腿造大腿环（必要时还有整腿
           环和"加长版膝盖环"）。
        4. 逐层解钟形 —— 每层拼一个 ``bellMatrix``，从 ramp 上取本层上下缘的
           半径比，调求解器，把结果顶圈存进 ``rows``。
        5. 拼曲面 —— 把 ``rows`` 按 Maya 的转置索引铺成控制点，配上 U/V knot，
           create 出 NURBS 曲面。

        中途有几处**只报警不中断**的诊断（bellScale 为零、腰髋距离异常、
        半径比爆炸），都是为了让"曲面炸了/塌了"这类无声故障能在 Script Editor
        里看到原因，而不是只能靠盯视口猜。
        """
        # 只处理自己的两个输出；其余（含基类的 locator 属性）交还给 Maya
        if plug.attribute() not in (SkirtBellCollider.attr_outputSurface,
                                    SkirtBellCollider.attr_outputRingMesh):
            return om.kUnknownParameter

        cls = SkirtBellCollider

        # --- 1. 读取输入值 ---------------------------------------------
        # 七个矩阵一律过 input_matrix：asMatrix() 返回的是临时 handle 的内部
        # 引用，语句一结束就悬空，必须当场拷一份（详见 utils.input_matrix）。
        inputBellMatrix = input_matrix(dataBlock, cls.attr_bellMatrix)
        leftHipMatrix = input_matrix(dataBlock, cls.attr_leftHipMatrix)
        leftKneeMatrix = input_matrix(dataBlock, cls.attr_leftKneeMatrix)
        leftHeelMatrix = input_matrix(dataBlock, cls.attr_leftHeelMatrix)
        rightHipMatrix = input_matrix(dataBlock, cls.attr_rightHipMatrix)
        rightKneeMatrix = input_matrix(dataBlock, cls.attr_rightKneeMatrix)
        rightHeelMatrix = input_matrix(dataBlock, cls.attr_rightHeelMatrix)

        skirtType = dataBlock.inputValue(cls.attr_skirtType).asShort()
        height = dataBlock.inputValue(cls.attr_height).asFloat()
        # 用 MPlug 读，不走 dataBlock —— 见 utils.plug_vector 的说明：
        # createPoint 建出来的属性上，dataBlock 那条路读出来是垃圾值，
        # 而且相邻属性会互相串。
        thisObj = self.thisMObject()
        ringScale = plug_vector(thisObj, cls.attr_ringScale)
        bellScale = plug_vector(thisObj, cls.attr_bellScale)

        if DEBUG:
            # 连属性名一起打出来。如果两行显示的名字相同，说明两个类属性指向了
            # 同一个属性 —— 那就是 initialize 里串了，不是读取的问题。
            try:
                ringName = om.MFnAttribute(cls.attr_ringScale).name()
                bellName = om.MFnAttribute(cls.attr_bellScale).name()
            except Exception:
                ringName = bellName = "（读不到属性名）"

            om.MGlobal.displayInfo(
                "skirtBellCollider.compute: %s=(%.3f, %.3f, %.3f)  "
                "%s=(%.3f, %.3f, %.3f)"
                % (ringName, ringScale.x, ringScale.y, ringScale.z,
                   bellName, bellScale.x, bellScale.y, bellScale.z))

            if ringName == bellName:
                om.MGlobal.displayError(
                    "skirtBellCollider: attr_ringScale 和 attr_bellScale "
                    "指向同一个属性（%s）—— initialize 里属性串了。" % ringName)

        # bellScale 读成零向量时，钟形矩阵三个轴行全是零、只剩平移，网格所有点
        # 塌到该层原点上，曲面缩成一条竖线 —— 而且不会有任何报错。直接喊出来。
        if bellScale.length() < 1e-9:
            om.MGlobal.displayError(
                "skirtBellCollider: bellScale 读出来是零向量，曲面会塌成一条线。"
                "（属性值本身可能是好的 —— 见 utils.plug_vector 里关于 "
                "dataBlock 那条路读不对的说明。）")
            return
        # 标量属性走 dataBlock 是安全的（只有 createPoint 建的 3 分量属性有坑）
        bellSubdivision = dataBlock.inputValue(cls.attr_bellSubdivision).asInt()
        ringSubdivision = dataBlock.inputValue(cls.attr_ringSubdivision).asInt()
        falloff = dataBlock.inputValue(cls.attr_falloff).asFloat()
        collision = dataBlock.inputValue(cls.attr_collision).asFloat()
        tightness = dataBlock.inputValue(cls.attr_tightness).asFloat()
        leftRingAxis = dataBlock.inputValue(cls.attr_leftRingAxis).asShort()
        rightRingAxis = dataBlock.inputValue(cls.attr_rightRingAxis).asShort()
        bellAxis = dataBlock.inputValue(cls.attr_bellAxis).asShort()

        # 渐变曲线属性
        # ramp 不是普通数据属性，读不到 dataBlock 里，只能用 MRampAttribute
        # 现场包一层；取值走 utils.ramp_value_at（API 1.0 是 out-param）。
        rampAttr = om.MRampAttribute(self.thisMObject(), cls.attr_bellScaleRamp)

        # --- 2. 量身 ------------------------------------------------------
        # 只取六个关节的**位置**，它们的旋转在造环时才用到
        # 左右对应关节的中点
        LH = taxis(leftHipMatrix)
        LK = taxis(leftKneeMatrix)
        LHe = taxis(leftHeelMatrix)
        RH = taxis(rightHipMatrix)
        RK = taxis(rightKneeMatrix)
        RHe = taxis(rightHeelMatrix)

        layout = computeSkirtLayout(inputBellMatrix, LH, LK, LHe, RH, RK, RHe,
                                    skirtType, height, bellAxis)

        L_thigh = layout.L_thigh
        L_calf = layout.L_calf
        dir_vector = layout.dir_vector
        levelDistances = layout.levelDistances
        h_val = layout.h_val
        N = layout.N

        # 裙子的起点，沿选定的腰部轴刚性对齐
        P_start = layout.W

        # bellMatrix 没连接时 W 落在世界原点，d_hip 就变成"髋部到原点"的距离，
        # 裙子会从原点沿轴拉出去很远 —— 曲面缩成一条极长的线。这里直接报出来，
        # 免得只能靠看视口猜。
        if layout.d_hip > (L_thigh + L_calf) * 2.0:
            om.MGlobal.displayWarning(
                "skirtBellCollider: 腰到髋的距离 %.1f 远大于整条腿长 %.1f —— "
                "bellMatrix 很可能没连接（默认单位矩阵会把腰放在世界原点）。"
                % (layout.d_hip, L_thigh + L_calf))

        # --- 3. 造碰撞环 ---------------------------------------------------
        # 大腿环：从髋部出发瞄准膝盖，Y 轴长 = 大腿长，X/Z 长 = 环半径。
        # 环是"髋→膝"这一整段，不是膝盖上的一个圈。
        thighScale = om.MVector(ringScale.x, L_thigh * ringScale.y, ringScale.z)
        leftHipToKnee = _createRingMatrix(leftHipMatrix, thighScale, leftRingAxis, LK)
        rightHipToKnee = _createRingMatrix(rightHipMatrix, thighScale, rightRingAxis, RK)

        # 下面四个只有长裙用得上；短裙留着单位矩阵，永远不会被放进 ringMatrices。
        # （单位矩阵当环用是没有意义的，所以千万不能"顺手"把它们也加进去。）
        leftHipToHeel = om.MMatrix()
        rightHipToHeel = om.MMatrix()
        leftHipToKneeExtended = om.MMatrix()
        rightHipToKneeExtended = om.MMatrix()
        if skirtType == 1:
            # 整条腿的长度。左右两个变量算的是同一个值（L_thigh/L_calf 本来就
            # 已经是左右平均），保留两份只是为了和 C++ 版逐行对应、并留出日后
            # 改成左右各算各的余地。
            leftLegLen = L_thigh + L_calf
            rightLegLen = L_thigh + L_calf
            leftHeelScale = om.MVector(ringScale.x, leftLegLen * ringScale.y, ringScale.z)
            rightHeelScale = om.MVector(ringScale.x, rightLegLen * ringScale.y, ringScale.z)

            # 整腿环：髋 → 踝，长度是整条腿。裙摆能垂到小腿，所以必须有它。
            leftHipToHeel = _createRingMatrix(leftHipMatrix, leftHeelScale, leftRingAxis, LHe)
            rightHipToHeel = _createRingMatrix(rightHipMatrix, rightHeelScale, rightRingAxis, RHe)

            # "加长版膝盖环"：方向瞄膝盖（所以屈膝时它跟着大腿走），但长度用的
            # 是**整条腿**的长度 —— 于是它从髋部沿大腿方向捅出去，一直伸过膝盖。
            # 这是 tightness 的另一半：加上它，裙摆会被大腿方向顶得更开（宽松）；
            # 去掉它，裙摆只受"髋→踝"这根环约束，贴得更紧（合体）。
            leftHipToKneeExtended = _createRingMatrix(leftHipMatrix, leftHeelScale, leftRingAxis, LK)
            rightHipToKneeExtended = _createRingMatrix(rightHipMatrix, rightHeelScale, rightRingAxis, RK)

        # 各层沿轴距离已经由 computeSkirtLayout 算好（levelDistances）

        # --- 4. 逐层求解 ---------------------------------------------------
        # rows[v] 是曲面 V 方向第 v 行的控制点。N 段钟形共用 N+1 条轮廓线：
        # 第 0 层贡献底圈和顶圈，其余各层只贡献顶圈（它们的底圈就是上一层的
        # 顶圈，重复取会多出一行、曲面会在那儿出现零厚度的褶）。
        rows = [None] * (N + 1)
        controlPoints = om.MPointArray()
        uKnots = om.MDoubleArray()
        vKnots = om.MDoubleArray()

        # 设置 uKnots：kPeriodic 形式下均匀分布（3 次）
        # 数量是 CV 数 + degree - 1 = (bellSubdivision + 3) + 3 - 1，即 +5；
        # 从 -2 起算是周期形式的惯例（前后各留出 degree-1 个"翻卷"节点），
        # 这样接缝处的连续性才对。
        numKnotsInU = bellSubdivision + 5
        for j in range(numKnotsInU):
            uKnots.append(float(j - 2))

        # ramp 的采样参数是 距离 / 总裙长，所以总裙长为 0 时要兜底防除零
        h_safe = 1e-5 if h_val < 1e-5 else h_val

        # 逐层求解每个钟形
        for i in range(N):
            # 本层的高度 = 相邻两层距离之差。夹一个下限：dist 为 0 会让
            # bellMatrix 的 Y 行变成零向量，求解器里"除以钟高"就炸了。
            dist = levelDistances[i + 1] - levelDistances[i]
            safeDist = 1e-4 if dist < 1e-4 else dist
            # 本层钟形的底心：从腰沿裙子方向走 levelDistances[i]
            P_bell = P_start + dir_vector * levelDistances[i]

            # 构建 bellMatrix 的朝向（对零值/未连接的输入做了充分防护）
            # 思路：Y 必须是裙子方向（不可协商），X 借腰部矩阵的 X 来定"正面"
            # 朝哪 —— 这样角色转身时裙子的 UV / CV 编号跟着转，不会打滑。
            # 三级回退：腰 X → 腰 Z 叉乘 → 世界 X。每一级都可能因为矩阵没连接
            # （单位矩阵）或与 dir_y 平行而退化，退化了就换下一级。
            dir_y = dir_vector

            raw_waist_X = xaxis(inputBellMatrix)
            waist_X = (om.MVector(1, 0, 0) if raw_waist_X.length() < 1e-4
                       else raw_waist_X.normal())

            raw_X = waist_X - dir_y * (waist_X * dir_y)
            if raw_X.length() < 1e-4:
                raw_waist_Z = zaxis(inputBellMatrix)
                waist_Z = (om.MVector(0, 0, 1) if raw_waist_Z.length() < 1e-4
                           else raw_waist_Z.normal())
                crossed = dir_y ^ waist_Z
                X = om.MVector(1, 0, 0) if crossed.length() < 1e-4 else crossed.normal()
            else:
                X = raw_X.normal()

            raw_Z = X ^ dir_y
            Z = om.MVector(0, 0, 1) if raw_Z.length() < 1e-4 else raw_Z.normal()

            # 按实际距离比例查询渐变曲线的值
            # ramp 是沿**裙子高度**采样的（不是沿角度）：参数 0 = 腰，
            # 1 = 裙摆，值就是该高度处的半径倍率（1 = 不缩放）。
            # 采样点用真实距离比例而不是层序号 i/N —— 各层高度并不相等，
            # 用序号会让 ramp 在视觉上被拉歪。
            t_bottom = float(levelDistances[i] / h_safe)
            t_top = float(levelDistances[i + 1] / h_safe)
            raw_scale_bottom = ramp_value_at(rampAttr, t_bottom)
            raw_scale_top = ramp_value_at(rampAttr, t_top)

            # 底可以为 0（裙子在这一层收成一个点，合法）；顶不行 —— 它是下面
            # bottomRatio 的分母，还要拿去缩放 X/Z 轴，为 0 会让矩阵退化。
            scale_bottom = raw_scale_bottom if raw_scale_bottom > 0.0 else 0.0
            scale_top = 1e-5 if raw_scale_top < 1e-5 else raw_scale_top

            # 用 bellScale 属性缩放各轴
            # 这里只把 scale_top 乘进 X/Z：求解器约定"顶圈半径 = 1（局部）"，
            # 所以本层的绝对半径全部由矩阵轴长承担，而底圈相对顶圈的比例
            # 另外通过 bellBottomRadius 传进去（见下面的 bottomRatio）。
            scaled_X = X * (bellScale.x * scale_top)
            # Y 行同时编码朝向和**本层高度** —— 这就是求解器里"高度固定传 1"
            # 却仍能得到正确尺寸的原因。
            Y = dir_y * (safeDist * bellScale.y)
            scaled_Z = Z * (bellScale.z * scale_top)

            bellMatrix = matrix_from_rows([
                [scaled_X.x, scaled_X.y, scaled_X.z, 0.0],
                [Y.x, Y.y, Y.z, 0.0],
                [scaled_Z.x, scaled_Z.y, scaled_Z.z, 0.0],
                [P_bell.x, P_bell.y, P_bell.z, 1.0]])

            def solveForRings(rings, wantBottom):
                """返回 (底圈 CV 或 None, 顶圈 CV)。

                定义成**嵌套函数**是有原因的：它要闭包捕获本层循环里刚算好的
                一大堆量（`bellMatrix`、`scale_bottom`/`scale_top`、`i`、以及
                各个属性），提出去就得传十几个参数。代价是每层都会重新定义一次
                函数对象 —— 层数只有 2~3，可以忽略。

                之所以要"同一层解多次"，是因为长裙最后一层要分别用**带膝盖环**
                和**不带膝盖环**各解一遍，再按 ``tightness`` 混合（见下面）。
                所以这里必须做成可重复调用的，而不是把求解写死在循环体里。

                `wantBottom` 只有第 0 层传 True：其余层的底圈等于上一层的顶圈。
                """
                inputs = BellColliderInputs()
                inputs.bellMatrix = bellMatrix
                inputs.ringMatrices = rings
                inputs.bellSubdivision = bellSubdivision
                inputs.ringSubdivision = ringSubdivision
                # 这个比值是整个节点最脆的一处：scale_top 的下限只有 1e-5，
                # 一旦 ramp 在这一层的顶部接近 0，比值就能冲到 1e5，钟形半径
                # 变成天文数字 —— 表现为曲面炸成一片延伸到天边的条纹。
                # 正常形状的底/顶半径比不会超过个位数，这里留足余量再夹死。
                bottomRatio = scale_bottom / scale_top
                if bottomRatio > _MAX_BOTTOM_RATIO:
                    om.MGlobal.displayWarning(
                        "skirtBellCollider: 第 %d 层的底/顶半径比达到 %.1f，"
                        "已夹到 %.0f。检查 bellScaleRamp 顶部是不是接近 0。"
                        % (i, bottomRatio, _MAX_BOTTOM_RATIO))
                    bottomRatio = _MAX_BOTTOM_RATIO
                inputs.bellBottomRadius = bottomRatio
                inputs.falloff = falloff
                inputs.collision = collision

                outputs = BellColliderOutputs()
                try:
                    BellColliderSolver.solve(inputs, outputs)
                except Exception as exc:
                    om.MGlobal.displayError(
                        "SkirtBellCollider: 第 %d 层钟形求解失败：%s" % (i, exc))
                    raise

                # 只要网格的点，不要它的拓扑 —— 本节点输出的是 NURBS 曲面，
                # 求解器返回的 mesh 在这里纯粹是"变形后顶点"的载体。
                meshFn = om.MFnMesh(outputs.outputBellMeshData)
                meshPoints = mesh_points(meshFn)

                bottom = _getCurvePoints(meshPoints, bellSubdivision, True) if wantBottom else None
                top = _getCurvePoints(meshPoints, bellSubdivision, False)

                return bottom, top

            # tightness 只作用于长裙的最后一层（i == 2），也就是裙摆那一圈 ——
            # 上面几层贴着髋和大腿，形状由解剖结构决定，没有"松紧"可言；
            # 只有垂在小腿附近的裙摆才有"包住腿"还是"敞开"的自由度。
            if skirtType == 1 and i == 2:
                # 只有整腿环：裙摆只需绕开小腿，能收得比较紧
                ringsWithoutKnee = [leftHipToHeel, rightHipToHeel]

                # 额外加上加长版膝盖环：大腿方向也来顶一把，裙摆被撑得更开
                ringsWithKnee = list(ringsWithoutKnee)
                ringsWithKnee.append(leftHipToKneeExtended)
                ringsWithKnee.append(rightHipToKneeExtended)

                topWith = None
                topWithout = None

                # 端点值时**跳过**其中一次求解：求解是这个节点最贵的一步，
                # tightness 拉到 0 或 1 时另一份结果反正用不上。
                # 也正因如此，下面取用前必须先判端点，不能直接 lerp（那一侧
                # 可能是 None）。
                if tightness < 1.0:
                    _, topWith = solveForRings(ringsWithKnee, False)
                if tightness > 0.0:
                    _, topWithout = solveForRings(ringsWithoutKnee, False)

                # 注意方向：tightness=0 → 带膝盖环（宽松/蓬），
                # tightness=1 → 不带（贴腿/紧）。名字和"带不带环"是反着的。
                if tightness <= 0.0:
                    rows[i + 1] = topWith
                elif tightness >= 1.0:
                    rows[i + 1] = topWithout
                else:
                    # 中间值：逐 CV 线性插值两份结果。两份 CV 数必然相同
                    # （同一个 bellSubdivision），所以可以直接一一对应。
                    blendedTop = om.MPointArray()
                    blendedTop.setLength(topWith.length())
                    for j in range(topWith.length()):
                        blendedTop.set(lerp_point(topWith[j], topWithout[j], tightness), j)
                    rows[i + 1] = blendedTop
            else:
                # 对应 i == 0 或 1
                # 上面几层：大腿环必给；长裙再补上整腿环 —— 屈膝时小腿会甩到
                # 大腿旁边，不带上它裙子会穿过小腿。
                bellRings = [leftHipToKnee, rightHipToKnee]
                if skirtType == 1:
                    bellRings.append(leftHipToHeel)
                    bellRings.append(rightHipToHeel)

                bottom, top = solveForRings(bellRings, i == 0)
                if i == 0:
                    rows[0] = bottom
                rows[i + 1] = top

        # --- 5. 拼成 NURBS 曲面 --------------------------------------------
        # 用实际的层距离填充 vKnots
        # V 方向是 1 次开放曲线，CV 数 = N+1，knot 数正好也是 N+1。
        # 这里**不用**均匀的 0,1,2...，而是直接填真实距离 —— 曲面的 V 参数
        # 于是就是"从腰量起的距离"，沿 V 做 uv/贴图或跟随时尺度是均匀的，
        # 不会因为各层高度不等而在中间那层被挤压。
        for i in range(N + 1):
            vKnots.append(levelDistances[i])

        # 按 Maya 要求的转置索引布局填充控制点：
        # index = (numCVsInV * uIndex) + vIndex
        # 也就是 U 为外层、V 为内层 —— 而 rows 是按 V（层）组织的，
        # 所以这里是一次**转置**，不能直接把 rows 首尾拼起来。
        numU = bellSubdivision + 3    # 含周期形式重复的那 3 个 CV
        numV = N + 1                  # 层数 + 1 条轮廓线
        controlPoints.setLength(numU * numV)
        for u in range(numU):
            for v in range(numV):
                # numU - 1 - u：U 方向**倒序**取。钟形顶点是按角度递增排的，
                # 倒过来取等于反转 U 的绕向，也就把曲面法线翻了个面。
                # 去掉这个倒序，裙子的正反面会整个反过来（单面材质从外面
                # 看不见）。C++ 版是同样的写法。
                controlPoints.set(rows[v][numU - 1 - u], u * numV + v)

        # 创建 NURBS 曲面
        # U 三次周期 = 横向一圈光滑闭合；V 一次开放 = 纵向逐层直连、
        # 层与层之间是折线。V 不做三次的原因：层数只有 2~3，三次会要求更多
        # CV，而且会让裙子在腰部和裙摆处"鼓"出去，反而不受控。
        surfaceData = om.MFnNurbsSurfaceData().create()

        surfaceFn = om.MFnNurbsSurface()
        surfaceFn.create(
            controlPoints,
            uKnots,
            vKnots,
            3,  # U 方向次数（三次，周期）
            1,  # V 方向次数（一次，开放）
            om.MFnNurbsSurface.kPeriodic,  # U 方向形式
            om.MFnNurbsSurface.kOpen,      # V 方向形式
            False,                         # 是否创建有理曲面
            surfaceData,
        )

        # 写入输出插槽
        outputHandle = dataBlock.outputValue(SkirtBellCollider.attr_outputSurface)
        outputHandle.setMObject(surfaceData)

        # 碰撞环合并成一个网格，供显示用。用的就是上面参与求解的那几个环矩阵，
        # 所以显示出来的一定和实际碰撞用的一致。
        ringMatrices = [leftHipToKnee, rightHipToKnee]
        if skirtType == 1:
            ringMatrices.append(leftHipToHeel)
            ringMatrices.append(rightHipToHeel)

        # 参数固定传 (axis=1, height=1)：环的真实朝向和长度已经全部编码在
        # ringMatrices 的轴长里了，makeRingsMesh 只需在局部空间造一个
        # "高 1、半径 1"的标准柱体，再由矩阵拉成实际尺寸。
        # 这里**不含**加长版膝盖环 —— 它只是 tightness 的内部手段，画出来
        # 会让人以为多了一根碰撞体。
        ringMesh = BellColliderSolver.makeRingsMesh(
            ringMatrices, 1, ringSubdivision, 1)
        dataBlock.outputValue(
            SkirtBellCollider.attr_outputRingMesh).setMObject(ringMesh)

        # 两个输出都已经算好，所以一起标干净（而不是只 clean 传进来的 plug）。
        # 少 clean 一个的话，Maya 拉另一个输出时会把整个 compute 再跑一遍。
        dataBlock.setClean(SkirtBellCollider.attr_outputSurface)
        dataBlock.setClean(SkirtBellCollider.attr_outputRingMesh)

def creator():
    """节点工厂：Maya 每次实例化 ``skirtBellCollider`` 时调用。

    和 :func:`initialize` 一起作为参数传给 ``MFnPlugin.registerNode``
    （在 ``colliders_Node.py`` 的 ``initializePlugin`` 里）。

    ``asMPxPtr`` 是必须的：它把 Python 对象的所有权移交给 Maya 的 C++ 侧。
    直接 ``return SkirtBellCollider()`` 会在函数返回后被 Python 回收，
    Maya 拿到野指针 —— 表现是创建节点就崩溃。
    """
    return ompx.asMPxPtr(SkirtBellCollider())


def initialize():
    """注册节点时调用一次（每次加载插件一次），建好全部属性和依赖关系。

    这是**类级**的一次性工作：属性对象存在 ``SkirtBellCollider.attr_xxx``
    上，被所有节点实例共享。所以这里不能引用任何具体实例。

    几个不显然的地方：

    * **function set 复用是有讲究的。** ``nAttr`` / ``eAttr`` / ``tAttr``
      这几个可以反复 ``create``，因为每次 create 都会重新绑定到新属性上。
      但 3 分量属性**不行** —— ``createPoint`` 复用同一个
      MFnNumericAttribute 会让属性互相串（改 bellScale 却动了 ringScale），
      所以 ``ringScale`` / ``bellScale`` 一律走 ``utils.create_numeric3``，
      它内部为每个属性新建一个 function set 并当场校验。
    * **属性创建顺序 = 属性编辑器里的显示顺序**，所以这里是按"输入在前、
      输出在后"排的。
    * ``setKeyable(True)`` 让属性出现在通道盒里，可以打关键帧。
    * 输出属性一律 ``setWritable(False)``（不许别人往里写）+
      ``setStorable(False)``（不存进场景文件 —— 每次打开重算即可，
      几何数据存进去只会让文件白白变大）。
    * 末尾的 ``attributeAffects`` 是 DG 的脏值传播规则：把**每个**输入连到
      **两个**输出上，任何输入改变都会让两个输出同时变脏、下次取值触发
      一次 ``compute``。漏掉某一条的后果是"改了属性视口没反应"，
      而且不会有任何报错。
    """
    nAttr = om.MFnNumericAttribute()
    mAttr = om.MFnMatrixAttribute()
    eAttr = om.MFnEnumAttribute()
    tAttr = om.MFnTypedAttribute()

    # 各关节矩阵
    # 七个属性长得一模一样，所以用循环 + setattr 批量建，避免七段复制粘贴。
    # 属性名和类成员名的对应关系是 "attr_" + 属性名 —— compute 里就是按这个
    # 约定去取的，改名字必须两边一起改。
    # 长名和短名传成同一个字符串：这几个属性不常在 MEL 里手打，短名没必要
    # 另起一个缩写。
    for name in ("bellMatrix", "leftHipMatrix", "leftKneeMatrix", "leftHeelMatrix",
                 "rightHipMatrix", "rightKneeMatrix", "rightHeelMatrix"):
        attr = mAttr.create(name, name)
        mAttr.setKeyable(True)
        ompx.MPxNode.addAttribute(attr)
        setattr(SkirtBellCollider, "attr_" + name, attr)

    # 裙子类型
    # 默认 1 = Long：C++ 版默认值即为此，改它会让旧场景的显示变样。
    SkirtBellCollider.attr_skirtType = eAttr.create("skirtType", "skirtType", 1)
    eAttr.addField("Short", 0)
    eAttr.addField("Long", 1)
    eAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(SkirtBellCollider.attr_skirtType)

    # 高度
    # 最小值给 0.01 而不是 0：0 会让最后一层高度为零、层与层重合。
    # 注意这只是 UI 限制，setAttr 仍能写进 0，所以 computeSkirtLayout 里
    # 还会再夹一次。
    SkirtBellCollider.attr_height = nAttr.create(
        "height", "height", om.MFnNumericData.kFloat, 1.0)
    nAttr.setMin(0.01)
    nAttr.setMax(1.0)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(SkirtBellCollider.attr_height)

    # 环的缩放
    # 默认 (0.5, 1, 0.5)：X/Z 是环的半径（腿的粗细），Y 是沿骨骼长度方向的
    # 倍率，取 1 表示环正好和骨头一样长。
    # 必须用 create_numeric3 —— 见函数开头 docstring 里关于属性串味的说明。
    SkirtBellCollider.attr_ringScale = create_numeric3(
        "ringScale", default=(0.5, 1.0, 0.5), keyable=True)
    ompx.MPxNode.addAttribute(SkirtBellCollider.attr_ringScale)

    # 钟形的缩放
    # 默认 (0.8, 1, 0.8)：X/Z 是裙子的胖瘦，Y 是每层高度的倍率。
    SkirtBellCollider.attr_bellScale = create_numeric3(
        "bellScale", default=(0.8, 1.0, 0.8), keyable=True)
    ompx.MPxNode.addAttribute(SkirtBellCollider.attr_bellScale)

    # 钟形细分数
    # 它直接决定曲面 U 方向的 CV 数（bellSubdivision + 3）和 knot 数
    # （bellSubdivision + 5）。下限 3 是构成一圈的最少点数。
    SkirtBellCollider.attr_bellSubdivision = nAttr.create(
        "bellSubdivision", "bellSubdivision", om.MFnNumericData.kInt, 16)
    nAttr.setMin(3)
    nAttr.setMax(64)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(SkirtBellCollider.attr_bellSubdivision)

    # 环的细分数
    # 只影响 outputRingMesh 的显示光滑度，**不参与碰撞计算** ——
    # 求解只用环矩阵。调它改变不了裙子的形状。
    SkirtBellCollider.attr_ringSubdivision = nAttr.create(
        "ringSubdivision", "ringSubdivision", om.MFnNumericData.kInt, 16)
    nAttr.setMin(3)
    nAttr.setMax(64)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(SkirtBellCollider.attr_ringSubdivision)

    # 衰减
    # -1..1，是一个**余弦阈值**而不是距离：顶点方向与环推来的方向夹角余弦
    # 大于它才参与变形。-1 = 整圈都受影响，越接近 1 影响范围越窄。
    SkirtBellCollider.attr_falloff = nAttr.create(
        "falloff", "falloff", om.MFnNumericData.kFloat, 0.0)
    nAttr.setMin(-1.0)
    nAttr.setMax(1.0)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(SkirtBellCollider.attr_falloff)

    # 碰撞强度
    # 径向挤压的混合系数（见 bellColliderSolver.ringPush）。默认 1 = 完全推出，
    # 不留穿透。注意它只在环**真的插进**裙子里时才有效果，相切时不动。
    SkirtBellCollider.attr_collision = nAttr.create(
        "collision", "collision", om.MFnNumericData.kFloat, 1.0)
    nAttr.setMin(0.0)
    nAttr.setMax(1.0)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(SkirtBellCollider.attr_collision)

    # 紧身程度
    # 本节点独有，**只对长裙的最后一层**（裙摆）生效，短裙上调它毫无反应。
    # 0 = 蓬松（额外用大腿方向的加长环把裙摆撑开），1 = 贴腿。
    # 默认 0.5，即两种解算结果各占一半。
    SkirtBellCollider.attr_tightness = nAttr.create(
        "tightness", "tightness", om.MFnNumericData.kFloat, 0.5)
    nAttr.setMin(0.0)
    nAttr.setMax(1.0)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(SkirtBellCollider.attr_tightness)

    # 钟形缩放渐变曲线（Ramp）
    # 沿裙子**高度**采样（0=腰，1=裙摆），值 = 该处的半径倍率。
    # 这里只能建出属性本身，**建不了默认条目** —— 条目属于节点实例的数据，
    # 由 postConstructor 补，而 Maya 之后还会自己再插一条默认斜坡，
    # 两者冲突时要用 resetBellScaleRamp() 清干净（见那两处的说明）。
    SkirtBellCollider.attr_bellScaleRamp = om.MRampAttribute.createCurveRamp(
        "bellScaleRamp", "bellScaleRamp")
    ompx.MPxNode.addAttribute(SkirtBellCollider.attr_bellScaleRamp)

    # 左腿轴、右腿轴与裙子轴
    # 三个枚举结构一样，同样批量建。默认 X / -X / Y 的由来：
    # 左右腿互为镜像所以取相反的横向轴；裙子沿腰关节的 Y 往下长。
    # 六个选项的含义见 _getAxis。**不要**把 ringAxis 选成沿骨骼方向的轴 ——
    # 那会撞上 _createRingMatrix 里的 180 度反向退化分支。
    for attrName, default in (("leftRingAxis", 0), ("rightRingAxis", 3), ("bellAxis", 1)):
        attr = eAttr.create(attrName, attrName, default)
        eAttr.addField("X", 0)
        eAttr.addField("Y", 1)
        eAttr.addField("Z", 2)
        eAttr.addField("-X", 3)
        eAttr.addField("-Y", 4)
        eAttr.addField("-Z", 5)
        eAttr.setKeyable(True)
        ompx.MPxNode.addAttribute(attr)
        setattr(SkirtBellCollider, "attr_" + attrName, attr)

    # 输出 NURBS 曲面（对用户可见）
    # 不可写 + 不存盘：几何是每次求值算出来的，存进场景文件毫无意义、
    # 只会让文件变大，而且下次打开还是要重算。
    SkirtBellCollider.attr_outputSurface = tAttr.create(
        "outputSurface", "outputSurface", om.MFnData.kNurbsSurface)
    tAttr.setWritable(False)
    tAttr.setStorable(False)
    ompx.MPxNode.addAttribute(SkirtBellCollider.attr_outputSurface)

    # 输出碰撞环网格（新增）—— 需要手动接到 mesh shape 上才看得见
    SkirtBellCollider.attr_outputRingMesh = tAttr.create(
        "outputRingMesh", "outputRingMesh", om.MFnData.kMesh)
    tAttr.setWritable(False)
    tAttr.setStorable(False)
    ompx.MPxNode.addAttribute(SkirtBellCollider.attr_outputRingMesh)

    # 建立属性依赖关系
    # 全连全：每个输入都影响两个输出。粒度粗但不会漏 —— 漏掉一条的症状是
    # "改了属性视口不更新"，且没有任何报错，非常难查；而多连一条只是偶尔
    # 多算一次，代价小得多。
    affects = (
        SkirtBellCollider.attr_bellMatrix, SkirtBellCollider.attr_leftHipMatrix,
        SkirtBellCollider.attr_leftKneeMatrix, SkirtBellCollider.attr_leftHeelMatrix,
        SkirtBellCollider.attr_rightHipMatrix, SkirtBellCollider.attr_rightKneeMatrix,
        SkirtBellCollider.attr_rightHeelMatrix, SkirtBellCollider.attr_skirtType,
        SkirtBellCollider.attr_height, SkirtBellCollider.attr_ringScale,
        SkirtBellCollider.attr_bellScale, SkirtBellCollider.attr_bellSubdivision,
        SkirtBellCollider.attr_ringSubdivision, SkirtBellCollider.attr_falloff,
        SkirtBellCollider.attr_collision, SkirtBellCollider.attr_tightness,
        SkirtBellCollider.attr_bellScaleRamp, SkirtBellCollider.attr_leftRingAxis,
        SkirtBellCollider.attr_rightRingAxis, SkirtBellCollider.attr_bellAxis,
    )
    for attr in affects:
        ompx.MPxNode.attributeAffects(attr, SkirtBellCollider.attr_outputSurface)
        ompx.MPxNode.attributeAffects(attr, SkirtBellCollider.attr_outputRingMesh)
