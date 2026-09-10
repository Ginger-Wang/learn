"""``sources/bellColliderSolver.h`` / ``bellColliderSolver.cpp`` 的 Python 移植版
（Maya Python API 1.0）。

求解器与 DG 节点相互独立：输入一个钟形（bell）矩阵和一组环（ring）矩阵，
输出变形后的钟形网格及其底部轮廓曲线。
"""

import math

import maya.OpenMaya as om

from utils import (Plane, findSphereLineIntersection, maxis, taxis,
                   copy_points, lerp_point, mesh_points)


class BellColliderInputs(object):
    """求解器的**全部输入**，对应 C++ 的 ``struct BellColliderInputs``。

    存在的意义是把"从 DG 节点读属性"和"算几何"彻底分开：节点（`bellCollider`、
    `bellColliderMulti`、`skirtBellCollider`）只负责把属性填进这个结构体，
    求解器不碰 dataBlock，因此可以脱离 Maya 节点单独测试
    （见 ``test_node.py``）。

    字段含义：

    * `bellMatrix` —— 钟形的世界矩阵。**平移是钟形底面圆心，Y 轴（第 1 行）
      既是钟形的朝向又是它的高度**（轴长 = 高），X/Z 轴长决定横向缩放。
    * `ringMatrices` —— 每个碰撞环的世界矩阵，语义同上（Y 轴为环的朝向，
      X/Z 轴长为环的半径，基础半径按 1 建模）。列表为空则不产生任何变形。
    * `bellSubdivision` —— 钟形一圈的分段数。它同时是顶点布局的关键常量：
      下标 0 是中心点，``1..bellSubdivision`` 是底圈，其后是顶圈；
      所以代码里到处出现的 ``range(bellSubdivision + 1, length)``
      含义都是"只遍历顶圈"（底圈和中心点固定不动）。
    * `ringSubdivision` —— 环网格一圈的分段数，只影响 ``outputRingMesh``
      的显示精度，不参与碰撞计算。
    * `bellBottomRadius` —— 钟形底圈相对顶圈的半径比（顶圈半径固定为 1），
      用来做出上宽下窄的裙摆形状。
    * `falloff` —— 变形的角度阈值，取值 -1..1。只有点相对环的方向的余弦
      大于它才参与变形；越大则受影响的范围越窄。
    * `collision` —— 环的径向挤压强度 0..1，见 :func:`ringPush`。
    """

    __slots__ = ("bellMatrix", "ringMatrices", "bellSubdivision",
                 "ringSubdivision", "bellBottomRadius", "falloff", "collision")

    def __init__(self):
        """给所有输入填上默认值（与 C++ 头文件里的默认成员初始值一致）。"""
        self.bellMatrix = om.MMatrix()
        self.ringMatrices = []
        self.bellSubdivision = 16
        self.ringSubdivision = 16
        self.bellBottomRadius = 0.8
        self.falloff = 0.0
        self.collision = 0.0


class BellColliderOutputs(object):
    """求解器的**全部输出**，对应 C++ 的 ``struct BellColliderOutputs``。

    两个成员都是 Maya 的**数据对象**（MObject），不是 function set，
    可以直接 ``outputHandle.setMObject(...)`` 交给节点的输出属性。

    :func:`BellColliderSolver.solve` 负责把它们填上，节点侧只管转交。
    """

    __slots__ = ("outputCurveData", "outputBellMeshData")

    def __init__(self):
        """初始化为空 MObject（``isNull()`` 为真），由 solve 填充。"""
        self.outputCurveData = om.MObject()       # NURBS 曲线数据对象
        self.outputBellMeshData = om.MObject()    # 变形后的钟形网格数据对象


def _wrapParam(param, minParam, maxParam):
    """把 `param` 按周期 ``[minParam, maxParam)`` 循环折回区间内。

    例如区间 0..10 时：12 → 2，-1 → 9。用 ``fmod`` 之后补一次正负修正，
    是为了让负数也落回区间（Python/C 的 fmod 会保留被除数的符号）。
    区间长度非正时无从折回，直接返回下界。

    **本函数当前并未被调用** —— 它是 C++ 里 ``wrapParam`` 的逐行移植，
    原版同样是死代码（见 ``README.md``）。保留下来只为让两边源码一一对应，
    方便日后对照与回填，不要以为它在参与计算。
    """
    rng = maxParam - minParam
    if rng <= 0.0:
        return minParam
    p = param - minParam
    p = math.fmod(p, rng)
    if p < 0.0:
        p += rng
    return p + minParam


def makeBellCurve(points, bellSubdivision, use_bottom=False):
    """从钟形网格的点里取出一圈，做成周期性 NURBS 曲线。

    这就是节点 ``outputCurve`` 的来源：变形后的钟形轮廓，可以拿去驱动
    毛发、布料的导引线，或者做后续的 loft/挤出。

    `use_bottom` 为 True 取底圈（下标 ``1..bellSubdivision-1``），
    否则取顶圈（下标 ``bellSubdivision+1`` 到末尾）。默认取顶圈，
    因为顶圈才是被 :func:`BellColliderSolver.deformPoints` 变形过的那一圈。
    公开（无下划线前缀）是为了给 ``bellColliderMulti`` 复用。

    曲线参数是 1 次（折线）、``kPeriodic``（周期闭合）。周期形式要求
    首尾 CV 重合，所以最后补了一个 ``cvs[0]`` 和一个递增的 knot；
    knot 序列就是 0,1,2,... 的均匀节点。
    """
    START = 1 if use_bottom else bellSubdivision + 1
    END = bellSubdivision if use_bottom else points.length()

    # 逐点搬进 CV 数组，同时生成均匀递增的节点值
    cvs = om.MPointArray()
    knots = om.MDoubleArray()
    for i in range(START, END):
        knots.append(float(cvs.length()))
        cvs.append(points[i])

    # 周期曲线闭合：末尾重复首个 CV，并补一个节点
    knots.append(knots[knots.length() - 1] + 1)
    cvs.append(cvs[0])

    curveData = om.MFnNurbsCurveData().create()

    curveFn = om.MFnNurbsCurve()
    # 参数顺序：cvs, knots, degree, form, create2D, createRational, parentOrOwner
    curveFn.create(cvs, knots, 1, om.MFnNurbsCurve.kPeriodic, False, False, curveData)
    return curveData


def appendBellFaces(polygonCounts, polygonConnects, base, numSides):
    """追加一个钟形的面拓扑（底面 numSides 个三角 + 侧面 numSides 个四边）。

    只管面、不管顶点，顶点索引统一加 `base` 偏移。这样既能配合
    :func:`appendBellVertices` 从头生成，也能直接套在**已经变形好**的点上
    （``bellColliderMulti`` 合并多段钟形时就是后者）。

    顶点布局约定：偏移 0 是中心点，1..numSides 是底圈，
    numSides+1..2*numSides 是顶圈。

    顶圈不封盖 —— 钟形是个开口的裙摆，只有底面（朝中心点收成扇形三角）
    和侧面。两个循环里的 ``if i == numSides - 1`` 都是**回绕**处理：
    最后一个面要接回这一圈的第一个顶点，才能闭合成筒。
    """
    # 底面：numSides 个三角扇面，公共顶点是中心点
    for i in range(numSides):
        polygonCounts.append(3)
        polygonConnects.append(base + 0)
        polygonConnects.append(base + i + 1)  # 偏移 0 是中心点
        polygonConnects.append(base + (1 if i == numSides - 1 else i + 2))

    # 侧面：numSides 个四边形，逐个连接底圈与顶圈的相邻两点
    for i in range(numSides):
        polygonCounts.append(4)
        polygonConnects.append(base + i + 1)
        polygonConnects.append(base + numSides + i + 1)
        polygonConnects.append(base + (numSides + 1 if i == numSides - 1
                                       else numSides + i + 2))
        polygonConnects.append(base + (1 if i == numSides - 1 else i + 2))


def appendBellVertices(vertexArray, matrix, axis, numSides,
                       height, bottomRadius, topRadius):
    """按 :func:`appendBellFaces` 约定的布局追加钟形的顶点。

    先在**局部空间**摆好一个标准锥台，再统一右乘 `matrix` 变到世界空间
    （Maya 是行向量约定，所以是 ``point * matrix`` 而不是 ``matrix * point``）。

    参数：

    * `axis` —— 中心轴取哪一根：0=X、1=Y、2=Z。求解器一律传 1（Y 轴），
      这个参数存在只是为了和 C++ 版保持接口一致。
    * `height` —— 顶圈相对底圈沿中心轴的偏移量。
    * `bottomRadius` / `topRadius` —— 两圈的半径。

    追加顺序必须是 中心点 → 底圈 → 顶圈，因为 :func:`appendBellFaces`
    的索引计算完全依赖这个顺序。两圈的角度取样一致（``i/numSides*2π``），
    这样底圈第 i 点与顶圈第 i 点在同一条母线上，侧面四边形才不会扭曲。
    """
    vertexArray.append(om.MPoint(0, 0, 0) * matrix)  # 偏移 0：底面圆心

    for i in range(numSides):  # 底圈
        rad = float(i) / numSides * 2 * math.pi
        x = bottomRadius * math.cos(rad)
        z = bottomRadius * math.sin(rad)

        if axis == 0:
            p = om.MPoint(0, x, z)
        elif axis == 1:
            p = om.MPoint(x, 0, z)
        else:
            p = om.MPoint(x, z, 0)

        vertexArray.append(p * matrix)

    for i in range(numSides):  # 顶圈
        rad = float(i) / numSides * 2 * math.pi
        x = topRadius * math.cos(rad)
        z = topRadius * math.sin(rad)

        if axis == 0:
            p = om.MPoint(height, x, z)
        elif axis == 1:
            p = om.MPoint(x, height, z)
        else:
            p = om.MPoint(x, z, height)

        vertexArray.append(p * matrix)


def _appendBell(vertexArray, polygonCounts, polygonConnects,
                matrix, axis, numSides, height, bottomRadius, topRadius):
    """往三个数组里追加一个钟形（中心顶点 + 底圈 + 顶圈）。

    :func:`appendBellVertices` 和 :func:`appendBellFaces` 的组合封装：
    先记下当前顶点数作为 `base` 偏移，再加顶点、加面。这样同一组数组里
    可以连续追加**多个**互不干扰的钟形 —— :meth:`BellColliderSolver.makeRingsMesh`
    就是靠反复调它把所有环合进一个 mesh 的。
    """
    base = vertexArray.length()  # 本次追加的顶点起始下标
    appendBellVertices(vertexArray, matrix, axis, numSides,
                       height, bottomRadius, topRadius)
    appendBellFaces(polygonCounts, polygonConnects, base, numSides)


def buildMeshData(vertexArray, polygonCounts, polygonConnects):
    """把顶点/面数组打包成一个 mesh 数据对象（MFnMeshData）。

    返回的是**数据对象**而非 DAG 节点，可以直接塞进节点的 mesh 输出属性；
    也可以再包一层 ``MFnMesh(meshObject)`` 就地改点（:meth:`solve` 里
    的 ``setPoints`` 正是这么干的）。

    顶点为空时返回空数据对象 —— MFnMesh.create(0, 0, ...) 会报错，
    而空 mesh 数据接到 mesh shape 上是合法的（显示为空）。
    没有环连进来时就走这条路，节点不会报错，只是看不到东西。
    """
    meshObject = om.MFnMeshData().create()

    if vertexArray.length() == 0:
        return meshObject

    meshFn = om.MFnMesh()
    # API 1.0 的 create 需要显式传顶点数和面数
    meshFn.create(vertexArray.length(), polygonCounts.length(), vertexArray,
                  polygonCounts, polygonConnects, meshObject)
    return meshObject


def ringPush(point, ringPlane, ring_translate, ringMatrixInverse, collision):
    """`collision` 属性的全部作用所在：点陷进环里时，沿径向把它推到环表面。

    对应 C++ ``deformPoints`` 末尾的 "ring collision" 那一段。

    做法：把点投影到环的横截面上，量它到环轴的距离 `vec_len`；再把这个向量
    变换到环的局部空间量一次，得到 `local_len`。两者的比值 `delta` 就是环在
    该方向上的**实际半径**（环矩阵的 X/Z 缩放有多大，半径就有多大；
    ``makeBellMesh`` 建环时用的基础半径是 1）。

    于是判据非常直白：``delta > vec_len`` 就是"点在环半径以内"，
    也就是穿透了；此时把它推到半径处，推移量再乘 `collision` 做混合。

    **注意临界情况**：点正好落在环表面上时 ``delta == vec_len``，判据不成立，
    不产生任何位移。所以如果环和钟形只是"刚好相切"，调 collision 是看不到
    任何变化的 —— 环必须真的插进钟形里才有效果。
    """
    vec = ringPlane.projectPoint(point) - ring_translate
    vec_len = vec.length()
    if vec_len <= 1e-5:
        return point

    local_len = (vec * ringMatrixInverse).length()
    delta = vec_len / local_len if local_len > 1e-5 else 1.0

    if delta <= vec_len:      # 点在环外（或恰好在表面），不动它
        return point

    return point + vec.normal() * ((delta - vec_len) * collision)


class BellColliderSolver(object):
    """纯计算的求解器，对应 C++ 的 ``class BellColliderSolver``。

    全部是 ``@staticmethod`` —— 它不持有任何状态，只是一组函数的命名空间
    （和 C++ 版的静态成员函数一一对应）。因为不依赖 MPxNode / dataBlock，
    可以在 Maya 里直接调来做单元测试。

    调用关系：:meth:`solve` 是唯一入口，内部依次调
    :meth:`makeBellMesh`（造基础网格）→ :meth:`deformPoints`（逐环变形）
    → :meth:`averageDisplacements`（融合多环结果）。
    :meth:`makeRingsMesh` 与主流程无关，只用于把碰撞环画出来看。
    """

    @staticmethod
    def makeBellMesh(matrix, axis, numSides, height=1, bottomRadius=1, topRadius=1):
        """生成一个封闭的锥体/圆柱体：一个中心顶点，加上底圈和顶圈。

        这是变形前的**基础钟形**。``bottomRadius != topRadius`` 时是锥台
        （裙摆），相等时是圆柱。参数含义见 :func:`appendBellVertices`。
        """
        vertexArray = om.MPointArray()
        polygonCounts = om.MIntArray()
        polygonConnects = om.MIntArray()

        _appendBell(vertexArray, polygonCounts, polygonConnects,
                    matrix, axis, numSides, height, bottomRadius, topRadius)

        return buildMeshData(vertexArray, polygonCounts, polygonConnects)

    @staticmethod
    def makeRingsMesh(matrices, axis, numSides, height=1,
                      bottomRadius=1, topRadius=1):
        """把多个环合并成**一个** mesh 数据对象。

        用于 ``outputRingMesh``：一个 mesh shape 就能显示全部碰撞环，
        不需要每个环各建一个物体。

        纯粹是**可视化辅助**，碰撞计算只用 `ringMatrices`，不读这个网格。
        所以 `numSides`（即 `ringSubdivision`）只影响显示光滑度，调它不会
        改变变形结果。环用的也是同一个钟形构造器，默认半径 1 ——
        这个 1 正是 :func:`ringPush` 推断环实际半径时的基准。
        """
        vertexArray = om.MPointArray()
        polygonCounts = om.MIntArray()
        polygonConnects = om.MIntArray()

        for matrix in matrices:
            _appendBell(vertexArray, polygonCounts, polygonConnects,
                        matrix, axis, numSides, height, bottomRadius, topRadius)

        return buildMeshData(vertexArray, polygonCounts, polygonConnects)

    @staticmethod
    def deformPoints(inputs, baseBellPoints, bellPlane):
        """每个环生成一份变形后的钟形点集，返回 MPointArray 的列表。

        这是整个求解器的**核心**。注意它**不做**多环合并 —— 每个环独立地
        对基础点集变形一份，N 个环得到 N 份完整点集；合并交给
        :meth:`averageDisplacements`。这样拆开的好处是每个环的影响可以
        单独计算，互不污染。

        参数：

        * `inputs` —— :class:`BellColliderInputs`。
        * `baseBellPoints` —— 未变形的钟形顶点（世界空间）。只读，
          每轮都 ``copy_points`` 一份再改（MPointArray 赋值是引用绑定）。
        * `bellPlane` —— 钟形底面所在平面（原点 = 钟形底心，法线 = 钟形 Y 轴）。
          由 :meth:`solve` 算好传进来，避免每个环重复构造。

        每个环的处理分两步，彼此独立：

        1. **倾斜/避让**（受 `falloff` 控制）：把环的轴线投影到钟形底面上，
           求它与单位球的交点，得到"钟形被顶到哪儿"和"环推到哪儿"两个
           碰撞点；两点之差构成一个绕环心的旋转，按每个顶点相对环方向的
           夹角加权应用上去。效果是钟形整体朝着远离环的方向偏摆。
        2. **径向挤压**（受 `collision` 控制）：见 :func:`ringPush`。

        全程只动顶圈（``bellSubdivision + 1`` 之后的下标）—— 底圈和中心点
        是裙摆的固定端，不参与变形。
        """
        bellMatrix = inputs.bellMatrix
        bellMatrixInverse = bellMatrix.inverse()
        bellSubdivision = inputs.bellSubdivision
        falloff = inputs.falloff
        collision = inputs.collision

        bell_translate = taxis(bellMatrix)
        bellAxis = maxis(bellMatrix, 1)  # Y 轴：既是朝向，长度又是钟形的高
        bellNormal = bellAxis.normal()

        bellPointsList = []

        for ringMatrix in inputs.ringMatrices:
            ringMatrixInverse = ringMatrix.inverse()

            ringDirection = maxis(ringMatrix, 1)  # Y 轴
            ring_translate = taxis(ringMatrix)
            ringNormal = ringDirection.normal()
            ringPlane = Plane(ring_translate, ringNormal)  # 环的横截面

            # 把环的位置和朝向压到钟形底面上：整个避让计算都在这个 2D 平面里
            # 谈"方向"，这样钟形沿自身高度方向的位置差异不会干扰权重判断。
            ring_translate_proj = bellPlane.projectPoint(taxis(ringMatrix))
            ringDirection_proj = bellPlane.projectVector(ringDirection)

            bellPoints = copy_points(baseBellPoints)

            # 环轴与钟形轴几乎平行时投影长度趋于 0，方向无从定义，跳过倾斜
            # 变形（此时只剩下面的径向挤压起作用）。
            if ringDirection_proj.length() > 1e-3:
                # 在钟形的**局部**空间里求射线与单位球的交点。局部空间下钟形
                # 顶圈半径恰好是 1，所以"与单位球相交"就等价于"打在钟形侧壁
                # 上"；半径取 1.001 是留一点余量，避免相切时算不出交点。
                hitPoints = findSphereLineIntersection(ring_translate_proj * bellMatrixInverse,
                                                       ringDirection_proj * bellMatrixInverse,
                                                       om.MPoint(0, 0, 0), 1.001)

                collisionPointBell = om.MPoint()   # 钟形侧壁上被环顶到的点
                collisionPointRing = om.MPoint()   # 该点应该被推到的目标位置
                if hitPoints.length() > 0:
                    # 变回世界空间，并抬到顶圈的高度（+bellAxis = 沿高度方向
                    # 平移一个钟形高度）
                    collisionPointBell = hitPoints[0] * bellMatrix + bellAxis

                    # 环可能是"倒装"的：轴朝下时要把方向翻过来，否则会往错的
                    # 一侧算目标点。
                    linePointCoeff = 1 if ringNormal * bellNormal > 0 else -1

                    # 从环心沿这个方向走到**环的表面**：delta 就是环在该方向上
                    # 的实际半径（原理同 ringPush —— 世界长度 / 局部长度，
                    # 即环矩阵在该方向的缩放量）。
                    ring_proj = ringPlane.projectVector(ringDirection_proj * linePointCoeff)
                    delta = 1.0
                    ring_proj_scaled = om.MVector(0, 0, 0)
                    ring_proj_len = ring_proj.length()
                    if ring_proj_len > 1e-5:
                        local_len = (ring_proj * ringMatrixInverse).length()
                        if local_len > 1e-5:
                            delta = ring_proj_len / local_len
                        ring_proj_scaled = ring_proj.normal() * delta
                    linePoint = ring_translate + ring_proj_scaled

                    # 目标点必须与 collisionPointBell 到环心等距（保持"长度"
                    # 不变，只改方向），所以拿这个距离当球半径，与过环表面、
                    # 沿环轴的那条直线求交。
                    sphereLinePoints = findSphereLineIntersection(
                        linePoint, ringDirection, ring_translate,
                        (collisionPointBell - ring_translate).length())

                    # 两个交点里只要环轴正方向那一侧的（另一侧是穿到环背面）
                    for k in range(sphereLinePoints.length()):
                        if (sphereLinePoints[k] - ring_translate) * ringDirection > 0:
                            collisionPointRing = sphereLinePoints[k]

                # 两个碰撞点沿钟形高度方向的差，除以钟高归一化。负值 =
                # 目标点比当前点更低 = 环确实挤进来了，需要变形。
                bellAxisLen = bellAxis.length()
                if bellAxisLen > 1e-5:
                    collisionDelta = (bellPlane.distance(collisionPointRing) -
                                      bellPlane.distance(collisionPointBell)) / bellAxisLen
                else:
                    collisionDelta = 0.0

                if collisionDelta < 0:
                    # 构造"绕环心旋转"的复合变换：先平移到环心（取逆 =
                    # 把点搬到以环心为原点的坐标系），再带着旋转平移回去。
                    rotationMatrixFn = om.MTransformationMatrix()
                    rotationMatrixFn.setTranslation(om.MVector(ring_translate), om.MSpace.kWorld)
                    rotateMatrixInverse = rotationMatrixFn.asMatrixInverse()

                    # 把 X 旋转到最终目标点
                    # 即：能把 collisionPointBell 转到 collisionPointRing 的那个
                    # 最短弧旋转。所有受影响的顶点都共用它。
                    quat = om.MQuaternion(collisionPointBell - ring_translate,
                                          collisionPointRing - ring_translate)
                    rotationMatrixFn.rotateBy(quat, om.MSpace.kTransform)
                    rotateMatrix = rotationMatrixFn.asMatrix()

                    # 钟形顶圈原本所在的平面，用作变形后的"天花板"
                    upperBellPlane = Plane(bell_translate + bellAxis, bellNormal)

                    # 钟形顶部的变形（下标从 bellSubdivision+1 起 = 只动顶圈）
                    for j in range(bellSubdivision + 1, bellPoints.length()):
                        # 顶点相对环心的方向，同样压到钟形底面上再比
                        bellPoint_proj = bellPlane.projectPoint(bellPoints[j])
                        offset_proj = (bellPoint_proj * bellMatrixInverse -
                                       ring_translate_proj * bellMatrixInverse)

                        # 取值范围 -1..1
                        # 两个方向的余弦：1 = 顶点正处在环推来的方向上，
                        # -1 = 在钟形的背面。以此作为衰减权重的原始值。
                        weight = offset_proj.normal() * (ringDirection_proj * bellMatrixInverse).normal()

                        if weight > falloff:
                            # 把 falloff..1 重映射到 0..1，保证阈值处平滑起步
                            # 而不是突然跳变。falloff==1 时除数为 0，退化成全权重。
                            divisor = 1.0 - falloff
                            weight = (weight - falloff) / divisor if divisor > 1e-5 else 1.0

                            # 绕环心施加旋转
                            rp = bellPoints[j] * rotateMatrixInverse * rotateMatrix

                            # 用上平面做约束
                            # 旋转会把点抬到原顶圈高度之上（裙摆被"甩"起来），
                            # 压回平面上，避免顶部翻出去。
                            p = rp
                            if upperBellPlane.distance(rp) > 0:
                                p = upperBellPlane.projectPoint(rp)

                            # 按权重在原位置与变形位置之间插值
                            bellPoints.set(lerp_point(bellPoints[j], p, weight), j)

            # 环的碰撞挤压（与上面的倾斜变形叠加，独立开关）
            if collision > 1e-5:
                for j in range(bellSubdivision + 1, bellPoints.length()):
                    bellPoints.set(
                        ringPush(bellPoints[j], ringPlane, ring_translate,
                                 ringMatrixInverse, collision), j)

            bellPointsList.append(bellPoints)

        return bellPointsList

    @staticmethod
    def averageDisplacements(bellSubdivision, baseBellPoints, bellPointsList):
        """按位移长度的平方为权重，把各个环产生的位移融合成一个。

        :meth:`deformPoints` 给出的是 N 份独立结果，这里逐顶点合成一个最终
        位置。做法分两半，缺一不可：

        * **方向**取加权平均，权重是各位移长度的平方占比。用平方而不是长度
          本身，是为了让推得狠的那个环明显主导方向 —— 一个环推 10、另一个推 1
          时，权重是 100:1 而不是 10:1，弱的那个几乎不会把方向拽偏。
        * **长度**直接取 `maxDist`（最大的那个位移），**不做平均**。这是关键：
          若长度也平均，两个环从相反方向夹住钟形时会互相抵消，看起来像没碰撞；
          取最大值保证"至少满足挤得最狠的那个环"，不会穿模。

        因此 `wp` 只取它的 `normal()`，它自身的模长被丢弃。

        与 :meth:`deformPoints` 一致，只处理顶圈；`bellPointsList` 为空
        （没有环）时 `total` 恒为 0，原样返回基础点集。

        ``bellColliderMulti`` / ``test_node.py`` 会跳过 solve 直接调它，
        所以这里不能依赖任何 solve 里的上下文。
        """
        outBellPoints = copy_points(baseBellPoints)

        for i in range(bellSubdivision + 1, baseBellPoints.length()):
            # 第一遍：累计权重分母（平方和）并记下最大位移量
            total = 0.0
            maxDist = 0.0
            for bellPoints in bellPointsList:
                vec = bellPoints[i] - baseBellPoints[i]
                d = vec.length()
                total += math.pow(d, 2)

                if d > maxDist:
                    maxDist = d

            if total > 0:
                # 第二遍：按平方权重把各位移向量加权求和，得到融合方向
                wp = om.MVector()
                for bellPoints in bellPointsList:
                    vec = bellPoints[i] - baseBellPoints[i]
                    d = math.pow(vec.length(), 2)
                    w = d / total

                    wp += vec * w

                # 方向用加权和，长度用最大值（相反方向不会互相抵消）
                if wp.length() > 1e-5:
                    outBellPoints.set(outBellPoints[i] + wp.normal() * maxDist, i)

        return outBellPoints

    @staticmethod
    def solve(inputs, outputs):
        """求解器的唯一入口：吃 `inputs`，把结果写进 `outputs`。

        完整流程：

        1. 从 `bellMatrix` 拆出底心、高度轴、底面平面（后面到处都要用）。
        2. :meth:`makeBellMesh` 造出未变形的基础钟形。注意高度参数固定传 1、
           顶圈半径固定传 1 —— 真正的尺寸已经编码在 `bellMatrix` 的轴长里了，
           所以局部空间下钟形永远是"高 1、顶圈半径 1"的标准形状，
           :meth:`deformPoints` 里"与单位球求交"这一手才成立。
        3. :meth:`deformPoints` 逐环变形 → :meth:`averageDisplacements` 融合。
        4. **就地**改基础网格的点（``setPoints``），所以 `bellMesh` 既是第 2 步
           的基础网格、又是最终输出 —— 不需要重新 create 一遍拓扑。
        5. :func:`makeBellCurve` 从变形后的顶圈抽出轮廓曲线。

        恒返回 True（C++ 版返回 MStatus，这里保留同样的调用形态，
        失败一律走异常）。
        """
        bellMatrix = inputs.bellMatrix
        bellSubdivision = inputs.bellSubdivision
        bellBottomRadius = inputs.bellBottomRadius

        # 钟形的几何参照：底心、高度轴、底面平面
        bell_translate = taxis(bellMatrix)
        bellAxis = maxis(bellMatrix, 1)  # Y 轴
        bellNormal = bellAxis.normal()
        bellPlane = Plane(bell_translate, bellNormal)

        # 基础网格：高度和顶圈半径都取 1，实际尺寸由 bellMatrix 承担
        bellMesh = BellColliderSolver.makeBellMesh(bellMatrix, 1, bellSubdivision,
                                                   1, bellBottomRadius, 1)
        bellMeshFn = om.MFnMesh(bellMesh)

        baseBellPoints = mesh_points(bellMeshFn)

        # 每个环各变形一份，再融合成一份
        bellPointsList = BellColliderSolver.deformPoints(inputs, baseBellPoints, bellPlane)

        outBellPoints = BellColliderSolver.averageDisplacements(
            bellSubdivision, baseBellPoints, bellPointsList)

        # 就地写回：拓扑不变，只挪点
        bellMeshFn.setPoints(outBellPoints)

        # 顶圈轮廓 → 输出曲线
        outCurve = makeBellCurve(outBellPoints, bellSubdivision)

        outputs.outputCurveData = outCurve
        outputs.outputBellMeshData = bellMesh

        return True
