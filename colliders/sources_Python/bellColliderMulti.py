"""``bellColliderMulti`` —— 由控制曲线驱动的钟形碰撞器（Maya Python API 1.0）。

和 ``bellCollider`` 的关键区别：**曲线是输入，不是输出。**

``bellCollider`` 的钟形形状是算出来的（底圈半径 + 顶圈半径），用户只能通过属性
间接影响。这里反过来：场景里放 N 条控制曲线，**每条曲线就是钟形的一圈**，
节点沿曲线采样得到网格顶点。于是

    拖动曲线的 CV / 移动缩放曲线  ->  bellMesh 立刻跟着变形

形状完全由曲线决定，所以这个节点没有 ``bellBottomRadius``、也没有
``bellScaleRamp`` —— 想要什么轮廓，直接把曲线摆成什么样。

**曲线负责形状，``bellSubdivision`` 负责精度**，两者互不干涉：曲线上放几个 CV
只影响手动调整的自由度，网格每圈有多少个点由 ``bellSubdivision`` 说了算。
各条曲线的 CV 数也不必相同。

拿到各圈顶点后，碰撞求解完全复用 ``BellColliderSolver``：顶点布局是
``[中心, 圈0, 圈1, ..., 圈N]``，求解器的变形循环正好从 ``bellSubdivision + 1``
开始（跳过中心点和最底下那圈，也就是"腰"固定不动），中间各圈自动全部参与碰撞。

**注意**：``outputBellMesh`` 是碰撞后的结果，而控制曲线是输入、不受碰撞影响。
所以环顶进来的时候，mesh 会被推开而曲线留在原处 —— 这是控制器的正常表现。

节点自己不能往场景里建东西，所以"创建时自动生成曲线和 mesh"由配套的
``bellColliderMulti_setup.py`` 负责。
"""

import maya.OpenMaya as om
import maya.OpenMayaMPx as ompx

from utils import Plane, maxis, taxis, mesh_points, matrix_from_rows
from bellColliderSolver import (BellColliderInputs, BellColliderSolver,
                                buildMeshData)


NODE_NAME = "bellColliderMulti"


def sampleCurveRing(curveObj, numSides):
    """沿控制曲线**等弧长**采样 `numSides` 个点，作为钟形的一圈顶点。

    采样而不是直接取 CV，好处有三个：

    * 网格细分由节点的 ``bellSubdivision`` 决定，和曲线有几个 CV 无关 ——
      曲线可以只放很少几个 CV 方便手动控制，网格照样可以很密。
    * 各条曲线的 CV 数不必相同，随便用什么曲线都能接进来。
    * 按弧长（而不是参数）取点，即使曲线被 rebuild 过、参数化不均匀，
      采样点在曲线上依然是均匀分布的。

    注意 degree 1 的折线曲线本身没有曲率，把 ``bellSubdivision`` 调高只会在
    直边上多插点，轮廓不会变圆滑；想要靠细分变平滑，曲线得是 degree 3。

    **关于坐标空间**：这里取的是 ``MSpace.kObject``，但返回的点其实是
    **世界坐标**。因为 ``inputCurve`` 该接的是曲线 shape 的
    ``worldSpace[0]``（setup 脚本就是这么连的），流过来的曲线数据已经把
    transform 烘进去了，此时"对象空间"就等于世界空间。
    要是有人误接了 ``local``（未经 transform 的局部空间），
    整个网格会跑到原点附近 —— 曲线本身看着没动，mesh 却飞了。

    另外 ``kWorld`` 在这儿反而**不能用**：传进来的只是一份游离的曲线数据
    对象，没有挂在任何 DAG 路径上，问它"世界空间"是没有意义的。

    只被 :meth:`BellColliderMulti.compute` 调用，每条输入曲线一次。
    """
    curveFn = om.MFnNurbsCurve(curveObj)

    points = om.MPointArray()
    params = _sampleParams(curveFn, numSides)
    if not params:
        return points   # 退化成一个点的曲线，交给调用方判空

    point = om.MPoint()
    for param in params:
        curveFn.getPointAtParam(param, point, om.MSpace.kObject)
        # getPointAtParam 是 out-param，每次都写进同一个对象里，必须拷贝
        points.append(om.MPoint(point))

    return points


def _sampleParams(curveFn, numSides):
    """算出采样用的参数序列，空列表表示曲线退化了。

    优先按**弧长**均匀取：这样即使曲线被 rebuild 过、参数化不均匀，
    采样点在曲线上依然是均匀分布的。

    ``length()`` / ``findParamFromLength()`` 万一在某个 Maya 版本上不可用，
    就退回按**参数**均匀取。退回方案对参数化不均匀的曲线分布会差一些，
    但总比让异常冒出去、整个 compute 中断（表现为"mesh 完全不更新"）要好。

    两条路都**只取 numSides 个点、不取第 numSides 个**（``range(numSides)``，
    即弧长比例 0/n、1/n … (n-1)/n）：曲线是周期闭合的，取到 1.0 就和 0.0
    重合了，会多出一个和首点同位的顶点，侧面四边形随之退化成零面积。

    弧长为 0（曲线塌成一点）或 ``numSpans() <= 0`` 时返回空列表，
    :func:`sampleCurveRing` 会把空数组原样返回，
    compute 里靠 ``points.length() == numSides`` 这一句把它过滤掉。
    """
    try:
        total = curveFn.length()
        if total <= 1e-9:
            return []
        return [curveFn.findParamFromLength(total * float(i) / numSides)
                for i in range(numSides)]
    except Exception as exc:
        om.MGlobal.displayWarning(
            "bellColliderMulti: 按弧长采样不可用（%s: %s），改用参数均匀采样。"
            % (type(exc).__name__, exc))

    spans = curveFn.numSpans()
    if spans <= 0:
        return []
    return [spans * float(i) / numSides for i in range(numSides)]


def ringCentroid(points):
    """一圈顶点的质心。

    两处用到它，用途不同但都要求"和曲线本身严格对齐"：

    * :func:`bellSpaceFromRings` —— 用底圈质心当参考空间的原点、
      用底圈到顶圈的质心连线当 Y 轴（钟形的中轴）。
    * :func:`buildRingMeshArrays` —— 用底圈质心当网格的中心顶点，
      这样底面封盖的扇形三角不会歪（那里为了少一次数组遍历，
      把同样的求和逻辑就地展开写了一遍，没有调本函数）。

    这里手动累加 x/y/z 而不是用 MVector 相加，是因为 API 1.0 的
    MPoint 之间不能直接相加（``MPoint + MPoint`` 无定义），而
    ``points[i]`` 拿到的是 MPoint。逐分量写最直白也最不容易踩坑。

    调用方保证 `points` 非空（compute 里已经把退化曲线过滤掉了），
    所以这里不做除零保护。
    """
    n = points.length()
    cx = cy = cz = 0.0
    for i in range(n):
        cx += points[i].x
        cy += points[i].y
        cz += points[i].z
    return om.MPoint(cx / n, cy / n, cz / n)


def bellSpaceFromRings(ringPoints):
    """由各圈顶点反推碰撞求解要用的参考空间。

    这一步是必须的，不能让用户随便接一个 locator 进来：求解器把钟形近似成
    **自身局部空间里半径 1、高度 1 的单位形状**（``findSphereLineIntersection``
    里那个 1.001 的球就是干这个的）。矩阵一旦和曲线围出来的实际几何对不上，
    碰撞位移的量级和方向就全错 —— 表现为整圈被压低、被拉成不圆的形状。

    所以直接按曲线来定：

    * 原点  = 底圈质心
    * Y 轴  = 底圈质心 -> 顶圈质心（长度就是钟形高度）
    * X/Z 轴长度 = 顶圈的平均半径（顶圈在局部空间里正好落到半径 1 上）

    这仍然是个**近似**：只有各圈都垂直于 Y 轴时，圈上每个点的局部高度才正好
    落在 0..1；圈斜着放的时候只有质心严格对齐。求解器本来就是拿单位球做粗略
    判定，这个精度足够 —— 关键是量级要对得上。
    """
    # 只看最底和最顶两圈：中间各圈怎么摆都不影响参考空间，
    # 它只是给求解器一个"量级和朝向对得上"的坐标系
    bottomCenter = ringCentroid(ringPoints[0])
    topCenter = ringCentroid(ringPoints[-1])

    # Y 轴 = 中轴，**不归一化** —— 它的长度就是钟形的高度，
    # 求解器正是靠这个轴长把碰撞位移换算回归一化尺度的
    Y = topCenter - bottomCenter
    if Y.length() < 1e-6:
        # 所有圈叠在同一高度（比如全被压平了），中轴无从定义。
        # 退回世界 Y 轴，至少矩阵不退化、不会传个奇异矩阵给 inverse()。
        Y = om.MVector(0.0, 1.0, 0.0)

    # X/Z 的轴长 = 顶圈的平均半径，这样顶圈在局部空间里正好落在半径 1 上，
    # deformPoints 里"与半径 1.001 的单位球求交"才对得上真实几何
    top = ringPoints[-1]
    radius = 0.0
    for i in range(top.length()):
        radius += (top[i] - topCenter).length()
    radius /= top.length()
    if radius < 1e-6:
        radius = 1.0   # 顶圈塌成一点，避免矩阵奇异（inverse() 会算不出来）

    # 任取一个垂直于 Y 的方向当 X；钟形是旋转体，X/Z 具体朝哪无所谓
    axis = Y.normal()
    seed = om.MVector(1.0, 0.0, 0.0)
    if abs(axis * seed) > 0.99:
        # 种子向量和中轴几乎平行时叉积会趋于零向量，normal() 结果无意义
        # （钟形立着时 Y 就是世界 Y，X 轴恰好是这种情况），换一根轴当种子
        seed = om.MVector(0.0, 0.0, 1.0)

    # 叉积两次凑出右手正交系：X ⊥ Y，Z ⊥ (Y, X)。
    # 注意 API 1.0 里 float * MVector 不可用，缩放一律写成 MVector * float。
    X = (seed ^ axis).normal() * radius
    Z = (axis ^ X.normal()).normal() * radius

    # Maya 矩阵是行主序 + 行向量约定：前三行是三个轴，第四行是平移。
    # 必须走 utils.matrix_from_rows —— API 1.0 不能用列表直接构造 MMatrix，
    # 而 createMatrixFromList 在部分版本上会**静默失败**（矩阵仍是单位矩阵，
    # 表现为整个曲面退化成一个平面且毫无报错），那个封装里带了自检和回退。
    return matrix_from_rows([[X.x, X.y, X.z, 0.0],
                             [Y.x, Y.y, Y.z, 0.0],
                             [Z.x, Z.y, Z.z, 0.0],
                             [bottomCenter.x, bottomCenter.y,
                              bottomCenter.z, 1.0]])


def buildRingMeshArrays(ringPoints):
    """由各圈顶点拼出网格的顶点/面数组。

    `ringPoints` 是从底到顶的若干 MPointArray，每圈点数必须相同。
    顶点布局与 ``makeBellMesh`` 保持一致（求解器依赖这个顺序）::

        0                                中心点（底圈的质心）
        1              .. numSides       圈 0（最底下，不参与碰撞变形）
        1 + numSides   .. 2*numSides     圈 1
        ...

    **这个布局是硬约定，不能改。** ``BellColliderSolver.deformPoints`` /
    ``averageDisplacements`` 的循环都是 ``range(bellSubdivision + 1, length)``
    起步，也就是"跳过下标 0 的中心点和整个圈 0，从圈 1 开始一直到最顶上"。
    :meth:`BellColliderMulti.compute` 传给求解器的 ``bellSubdivision``
    正是这里的 `numSides`（每圈的点数），所以：

    * 中心点 + 圈 0 = 钟形的"腰"，固定不动（裙摆的固定端）；
    * 圈 1 及以上全部自动参与碰撞，**不需要为多圈写任何额外代码** ——
      这就是为什么本节点能白拿 ``bellCollider`` 的整套求解逻辑。

    为什么不复用 ``bellColliderSolver.appendBellFaces``：那个函数的拓扑是
    "底面扇形 + **一层**侧面四边形"，只对两圈（底圈 + 顶圈）成立。这里是
    任意 N 圈，侧面要铺 ``N-1`` 层，所以面拓扑必须在这儿自己生成。
    顶点布局倒是刻意保持一致，这样求解器不用改一行。

    与 ``makeBellMesh`` 的另一个区别：那边先在局部空间摆标准锥台再乘矩阵，
    这里的顶点是**曲线采样出来的世界坐标**，直接就位、不做任何变换 ——
    形状完全由曲线决定，节点不参与造型。

    两个循环里的 ``nxt = 0 if i == numSides - 1 else i + 1`` 都是**回绕**：
    每圈的最后一个面要接回该圈的第 0 个点，才能闭合成筒。
    顶圈不封盖 —— 钟形是个开口的裙摆。
    """
    ringCount = len(ringPoints)
    numSides = ringPoints[0].length()

    vertexArray = om.MPointArray()
    polygonCounts = om.MIntArray()
    polygonConnects = om.MIntArray()

    # 中心点取底圈的质心，这样底面封盖不会歪
    cx = cy = cz = 0.0
    bottom = ringPoints[0]
    for i in range(numSides):
        cx += bottom[i].x
        cy += bottom[i].y
        cz += bottom[i].z
    vertexArray.append(om.MPoint(cx / numSides, cy / numSides, cz / numSides))

    # 各圈顶点按"从底到顶"的顺序连续追加。追加顺序就是索引顺序，
    # 下面算面索引时全靠它 —— 圈 k 的第 i 点固定落在 1 + k*numSides + i。
    for points in ringPoints:
        for i in range(numSides):
            vertexArray.append(points[i])

    # 底面：中心点扇出到圈 0
    # 只有底面封盖（下标 0 是唯一的公共顶点），圈 0 的偏移量是 1
    for i in range(numSides):
        nxt = 0 if i == numSides - 1 else i + 1  # 回绕：最后一片接回第 0 点
        polygonCounts.append(3)
        polygonConnects.append(0)
        polygonConnects.append(1 + i)
        polygonConnects.append(1 + nxt)

    # 侧面：相邻两圈之间一层四边形
    # 一共铺 ringCount-1 层。每层的两个基址都是"1（跳过中心点）+ 圈号*每圈点数"，
    # 相邻两层共用同一圈顶点（上一层的 lower 就是下一层的 upper 的下一圈），
    # 所以层与层之间天然是**同一批顶点**、不会出现接缝或重复点。
    for k in range(ringCount - 1):
        lower = 1 + k * numSides         # 本层下沿：圈 k 的起始下标
        upper = 1 + (k + 1) * numSides   # 本层上沿：圈 k+1 的起始下标
        for i in range(numSides):
            nxt = 0 if i == numSides - 1 else i + 1  # 回绕，同上
            # 四边形按 下-上-上(next)-下(next) 绕行，保证法线朝外且各层一致
            polygonCounts.append(4)
            polygonConnects.append(lower + i)
            polygonConnects.append(upper + i)
            polygonConnects.append(upper + nxt)
            polygonConnects.append(lower + nxt)

    return vertexArray, polygonCounts, polygonConnects


class BellColliderMulti(ompx.MPxLocatorNode):
    """``bellColliderMulti`` 的 DG 节点本体（本 Python 版新增，C++ 版没有）。

    继承 ``MPxLocatorNode`` 而不是 ``MPxNode``，纯粹是为了和另外三个碰撞器
    节点保持一致（都能在视口里选中、有 transform 父节点）。本版本没有任何
    视口绘制（原因见 ``README.md``：API 1.0 无 drawOverride 绑定、legacy
    ``draw()`` 在 Viewport 2.0 下不可靠），所以 locator 只剩"可选中"这一个
    作用，几何一律通过 ``outputBellMesh`` / ``outputRingMesh`` 输出到真实的
    mesh shape 上。

    职责边界：本类**只负责读属性、拼网格、转交结果**，一点几何数学都不做 ——
    碰撞求解全部委托给 ``BellColliderSolver``（见 :meth:`compute`）。

    这些 ``attr_*`` 类属性是 Maya 属性的句柄（MObject），由 :func:`initialize`
    在插件注册时**一次性**填好，之后 :meth:`compute` 用它们去 dataBlock 里
    取值。它们是**类级**共享的：同一个节点类型的所有实例共用同一套属性定义，
    这是 Maya API 的标准写法，不是笔误。
    """

    typeId = om.MTypeId(0x01024)   # 接在其它三个节点之后

    # 没有 bellMatrix —— 碰撞的参考空间由 bellSpaceFromRings() 从曲线算出来
    attr_ringMatrix = om.MObject()
    attr_inputCurve = om.MObject()       # 数组：控制曲线，从底到顶
    attr_bellSubdivision = om.MObject()
    attr_ringSubdivision = om.MObject()
    attr_falloff = om.MObject()
    attr_collision = om.MObject()
    attr_outputBellMesh = om.MObject()
    attr_outputRingMesh = om.MObject()

    def __init__(self):
        """节点实例的构造函数，由 :func:`creator` 在每次 ``createNode`` 时调用。

        **必须显式调基类构造函数**：Python API 1.0 的 MPx* 基类不会自动初始化，
        漏掉这一句会导致节点底层的 C++ 对象没建起来，之后随便调个方法就崩
        Maya（不是抛异常，是直接闪退）。

        这里没有任何自己的状态 —— 节点是无状态的，每次 :meth:`compute` 都从
        dataBlock 重新读全部输入、重新算。所以不需要缓存、不需要脏标记管理。
        """
        ompx.MPxLocatorNode.__init__(self)

    def compute(self, plug, dataBlock):
        """DG 求值入口：由 Maya 在下游请求输出网格时自动调用。

        调用时机：任何一个 :func:`initialize` 里用 ``attributeAffects`` 声明过的
        输入变脏（曲线的 CV 被拖动、环 locator 被移动、``bellSubdivision``
        被改），Maya 就把两个输出标脏；等到有人真的要读输出时（mesh shape
        要刷新、渲染、或脚本读属性）才回调这里。**不是每帧都调**，是按需的。

        完整流程（六步）：

        1. 过滤 plug —— 只认识两个输出属性，其它一律交还给基类。
        2. 读 ``bellSubdivision``，得到每圈要采样多少个点。
        3. ``inputCurve[]`` 逐条 :func:`sampleCurveRing` → 各圈顶点（世界坐标）。
        4. ``ringMatrix[]`` 逐个读出碰撞环矩阵。
        5. :func:`buildRingMeshArrays` + ``buildMeshData`` 拼出未变形的网格，
           然后 :func:`bellSpaceFromRings` 反推参考空间，交给求解器变形，
           最后 ``setPoints`` **就地**改点（拓扑不变，不重建）。
        6. 把两个网格数据对象塞进输出并 ``setClean``。

        关键点：这里**不调** ``BellColliderSolver.solve``。solve 会自己
        ``makeBellMesh`` 造一个标准两圈锥台，那是 ``bellCollider`` 的形状逻辑，
        和"形状由曲线决定"直接冲突。所以本节点绕过 solve，只借它内部的
        :meth:`~BellColliderSolver.deformPoints` +
        :meth:`~BellColliderSolver.averageDisplacements` 两步 —— 这也正是那两个
        方法被写成不依赖 solve 上下文的静态方法的原因。

        另外本节点**不输出 ``outputCurve``**：曲线在这儿是输入，再输出一条
        轮廓曲线毫无意义，所以 ``makeBellCurve`` 在本文件里没有用到。

        提前 ``return``（不返回 kUnknownParameter）的几处都表示"输入还不成形，
        本次不更新输出"。注意此时**没有** ``setClean``，输出保持脏状态，
        下次请求还会再算一遍 —— 这正是想要的：等用户把曲线接够了自然就好了。
        """
        cls = BellColliderMulti

        # 只处理自己的两个输出；其余插槽交还基类（返回 kUnknownParameter 是
        # MPxNode 的约定，表示"这个 plug 我不管"）
        attr = plug.attribute()
        if attr not in (cls.attr_outputBellMesh, cls.attr_outputRingMesh):
            return om.kUnknownParameter

        # 网格每圈的点数由这个属性决定，与曲线的 CV 数无关
        numSides = dataBlock.inputValue(cls.attr_bellSubdivision).asInt()
        if numSides < 3:
            # 兜底：属性上已经 setMin(3)，但直接 setAttr / 读旧场景仍可能
            # 拿到更小的值，而少于 3 个点围不出闭合的圈，MFnMesh.create 会报错
            numSides = 3

        # --- 控制曲线 -> 各圈顶点 ---
        curveHandle = dataBlock.inputArrayValue(cls.attr_inputCurve)
        curveCount = curveHandle.elementCount()
        if curveCount < 2:
            return  # 至少要两圈才能围出侧面

        # elementCount 只数**已连接/已设值**的元素，逻辑索引可能不连续
        # （比如只接了 inputCurve[0] 和 inputCurve[5]），所以下面必须用
        # jumpToArrayElement 按物理索引遍历，不能用 jumpToElement（逻辑索引）。
        # 顶点圈的上下顺序 = 这里的遍历顺序 = 逻辑索引从小到大，所以
        # inputCurve[0] 必须是最底下那圈。
        ringPoints = []
        for i in range(curveCount):
            curveHandle.jumpToArrayElement(i)  # 按物理索引遍历
            curveObj = curveHandle.inputValue().asNurbsCurve()
            if curveObj.isNull():
                continue  # 插槽有值但曲线是空的（上游被删了），跳过
            try:
                points = sampleCurveRing(curveObj, numSides)
            except Exception as exc:
                # 采样失败就报出来。若让异常冒出去，整个 compute 会中断，
                # 表现为"改了曲线但 mesh 纹丝不动"，反而不好查。
                om.MGlobal.displayError(
                    "bellColliderMulti: 第 %d 条曲线采样失败：%s: %s"
                    % (i, type(exc).__name__, exc))
                continue
            if points.length() == numSides:   # 跳过退化成一个点的曲线
                ringPoints.append(points)

        if len(ringPoints) < 2:
            om.MGlobal.displayWarning(
                "bellColliderMulti: 只有 %d 圈有效顶点（至少要 2 圈），"
                "本次不更新网格。用 diagnose_multi.run() 看具体原因。"
                % len(ringPoints))
            return

        # --- 环矩阵 ---
        # 环的顺序无所谓：求解器对每个环独立算一份位移，再用
        # averageDisplacements 融合，不依赖数组次序。
        ringMatrixHandle = dataBlock.inputArrayValue(cls.attr_ringMatrix)
        ringCount = ringMatrixHandle.elementCount()
        ringMatrices = []
        for i in range(ringCount):
            ringMatrixHandle.jumpToArrayElement(i)  # 同样按物理索引
            # 必须拷贝：asMatrix() 返回的是 handle 内部引用
            # 这就是 utils.input_matrix() 存在的理由（那里有详细解释）：
            # inputValue() 是临时 handle，语句一结束就回收，引用随之悬空，
            # 之后读到的是被复用的内存 —— 表现为矩阵时对时错、轴变成零向量、
            # 坐标爆到十万量级，且**毫无报错**。这里用 MMatrix 的拷贝构造
            # 当场存下来；数组元素没法直接套 input_matrix（它按属性取值、
            # 不接受数组 handle），所以手写同样的一句。
            ringMatrices.append(
                om.MMatrix(ringMatrixHandle.inputValue().asMatrix()))

        # 这三个是普通标量属性（kInt / kFloat），走 dataBlock 读是安全的。
        # 只有 utils.create_numeric3 建出来的 3 分量属性才必须改走
        # utils.plug_vector —— 本节点没有那种属性，所以全程不需要 plug_vector。
        ringSubdivision = dataBlock.inputValue(cls.attr_ringSubdivision).asInt()
        falloff = dataBlock.inputValue(cls.attr_falloff).asFloat()
        collision = dataBlock.inputValue(cls.attr_collision).asFloat()

        # --- 建网格 ---
        # 先造出**未变形**的网格：顶点就是曲线采样点，拓扑按顶点布局约定生成。
        # 下面的碰撞变形是就地改点，所以这一个 bellMesh 既是基础网格
        # 又是最终输出，不会重建第二遍拓扑。
        bellMesh = buildMeshData(*buildRingMeshArrays(ringPoints))

        # --- 碰撞变形（没有环就直接输出原始形状）---
        if ringMatrices:
            bellMeshFn = om.MFnMesh(bellMesh)
            basePoints = mesh_points(bellMeshFn)

            # 参考空间由曲线自己算出来，保证和实际几何严格对齐
            bellMatrix = bellSpaceFromRings(ringPoints)

            inputs = BellColliderInputs()
            inputs.bellMatrix = bellMatrix
            inputs.ringMatrices = ringMatrices
            # 求解器靠这个值跳过中心点和底圈，这里就是每圈的点数
            inputs.bellSubdivision = numSides
            inputs.ringSubdivision = ringSubdivision
            inputs.falloff = falloff
            inputs.collision = collision

            # 钟形底面平面（原点 = 底圈质心，法线 = 中轴方向）。solve 内部
            # 本来会自己算这个，绕过 solve 就得在这儿补上。
            bellPlane = Plane(taxis(bellMatrix), maxis(bellMatrix, 1).normal())

            # 每个环独立变形一份 → 融合成一份。numSides 当 bellSubdivision 传，
            # 求解器据此跳过中心点和圈 0（"腰"固定不动）。
            pointsList = BellColliderSolver.deformPoints(
                inputs, basePoints, bellPlane)
            outPoints = BellColliderSolver.averageDisplacements(
                numSides, basePoints, pointsList)

            # 就地写回：拓扑不变，只挪点
            bellMeshFn.setPoints(outPoints)

        dataBlock.outputValue(cls.attr_outputBellMesh).setMObject(bellMesh)

        # 环的显示网格：纯可视化，碰撞计算只用 ringMatrices，不读这个网格。
        # 基础半径固定传 1 —— ringPush 就是拿"世界长度 / 局部长度"反推环的
        # 实际半径的，那个基准 1 必须和这里建环用的半径一致。
        # 没有环时这里得到的是空 mesh 数据（合法，显示为空），所以这两句
        # 放在 if 外面无条件执行：环被删掉后显示网格也要跟着清空。
        ringMesh = BellColliderSolver.makeRingsMesh(
            ringMatrices, 1, ringSubdivision, 1)
        dataBlock.outputValue(cls.attr_outputRingMesh).setMObject(ringMesh)

        # 告诉 DG 这两个输出已经算好了，否则 Maya 会认为还脏、反复回调 compute
        dataBlock.setClean(cls.attr_outputBellMesh)
        dataBlock.setClean(cls.attr_outputRingMesh)


def creator():
    """节点工厂函数，注册时交给 ``MFnPlugin.registerNode``。

    每次 ``createNode bellColliderMulti`` 时由 Maya 调用一次。

    ``asMPxPtr`` 是必须的、不能省：它把 Python 对象的所有权移交给 Maya 的
    C++ 侧。少了这一层，Python 的垃圾回收会在函数返回后把实例回收掉，而
    Maya 手里还留着指针 —— 结果是随机崩溃（常常不在创建的当口，而是过一会儿
    求值时才炸，极难定位）。
    """
    return ompx.asMPxPtr(BellColliderMulti())


def initialize():
    """定义节点的全部属性和依赖关系，注册时交给 ``registerNode``。

    在插件加载期间被 Maya 调用**一次**（见 ``colliders_Node.py`` 的
    ``initializePlugin``），比任何节点实例的诞生都早。它做三件事：

    1. ``创建属性`` → 存进 :class:`BellColliderMulti` 的 ``attr_*`` 类属性；
    2. ``addAttribute`` → 把属性挂到节点类型上；
    3. ``attributeAffects`` → 声明"哪个输入变了会让哪个输出变脏"，
       这是 DG 能自动重算的**唯一**依据。漏掉一条的后果是"改了属性但网格
       不更新"，而且不会有任何报错。

    末尾那个双层循环就是把 6 个输入 × 2 个输出全连一遍（12 条），
    比手写 12 行更不容易漏。

    注意 function set 的复用：这里的 `nAttr` / `tAttr` 被多个 ``create``
    调用共享，对 ``MFnNumericAttribute.create`` / ``MFnTypedAttribute.create``
    来说是安全的（每次 create 都会重新绑定到新属性）。**只有
    ``createPoint``（3 分量属性）不能共享 function set** —— 那会让属性串味，
    详见 ``utils.create_numeric3`` 的说明。本节点没有 3 分量属性，所以共用无碍。
    """
    nAttr = om.MFnNumericAttribute()
    mAttr = om.MFnMatrixAttribute()
    tAttr = om.MFnTypedAttribute()

    cls = BellColliderMulti

    # ringMatrix / inputCurve 是本节点最主要的两个输入，要能在节点编辑器的
    # 完整展开视图里直接看到并连线，所以刻意**不**设 hidden
    # （C++ 版把矩阵设成 hidden，是因为那边纯内部使用，用法不一样）。
    #
    # 另外两个设置是为了让数组端口好连：
    #   readable=False     —— 纯输入，不需要往外读；也是 indexMatters 的前提
    #   indexMatters=False —— connectAttr 可以不写索引，Maya 自动往后排，
    #                         在节点编辑器里直接拖线连过来就能用
    cls.attr_ringMatrix = mAttr.create("ringMatrix", "ringMatrix")
    mAttr.setArray(True)
    mAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(cls.attr_ringMatrix)

    # 控制曲线：索引 0 是最底下那圈，依次往上
    cls.attr_inputCurve = tAttr.create(
        "inputCurve", "inputCurve", om.MFnNurbsCurveData.kNurbsCurve)
    tAttr.setArray(True)
    tAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(cls.attr_inputCurve)

    # 网格每圈采样多少个点。和曲线的 CV 数无关，调它就能改网格精度。
    cls.attr_bellSubdivision = nAttr.create(
        "bellSubdivision", "bellSubdivision", om.MFnNumericData.kInt, 16)
    nAttr.setMin(3)
    nAttr.setSoftMax(64)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(cls.attr_bellSubdivision)

    cls.attr_ringSubdivision = nAttr.create(
        "ringSubdivision", "ringSubdivision", om.MFnNumericData.kInt, 16)
    nAttr.setMin(3)
    nAttr.setSoftMax(64)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(cls.attr_ringSubdivision)

    cls.attr_falloff = nAttr.create(
        "falloff", "falloff", om.MFnNumericData.kFloat, 0)
    nAttr.setMin(-1)
    nAttr.setMax(1)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(cls.attr_falloff)

    cls.attr_collision = nAttr.create(
        "collision", "collision", om.MFnNumericData.kFloat, 0)
    nAttr.setMin(0)
    nAttr.setMax(1)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(cls.attr_collision)

    # 两个输出都是 mesh 数据，接到普通 mesh shape 的 inMesh 上就能看见。
    #   writable=False —— 输出属性，禁止往里连线/setAttr
    #   storable=False —— 不存进 .ma 文件；每次打开场景重新算，
    #                     否则存一份陈旧的网格进去，打开时可能和曲线不一致
    cls.attr_outputBellMesh = tAttr.create(
        "outputBellMesh", "outputBellMesh", om.MFnData.kMesh)
    tAttr.setWritable(False)
    tAttr.setStorable(False)
    ompx.MPxNode.addAttribute(cls.attr_outputBellMesh)

    cls.attr_outputRingMesh = tAttr.create(
        "outputRingMesh", "outputRingMesh", om.MFnData.kMesh)
    tAttr.setWritable(False)
    tAttr.setStorable(False)
    ompx.MPxNode.addAttribute(cls.attr_outputRingMesh)

    # 全部输入 × 全部输出，无脑全连。理论上有几对是多余的
    # （比如 ringSubdivision 只影响 outputRingMesh、falloff 只影响 bellMesh），
    # 但多声明一条依赖只是让 compute 偶尔多跑一次，漏声明一条却会导致
    # "改了属性网格不动"这种无声故障 —— 宁滥毋缺。
    drivers = (cls.attr_ringMatrix, cls.attr_inputCurve,
               cls.attr_bellSubdivision, cls.attr_ringSubdivision,
               cls.attr_falloff, cls.attr_collision)

    outputs = (cls.attr_outputBellMesh, cls.attr_outputRingMesh)

    for driver in drivers:
        for output in outputs:
            ompx.MPxNode.attributeAffects(driver, output)
