"""
``bellCollider`` 节点。与 C++ 版的关键差异：**不做视口绘制**。

C++ 版靠 ``MPxDrawOverride`` 把钟形和碰撞环画在视口里（不是场景物体）。
API 1.0 没有 drawOverride 绑定，而 legacy ``draw()`` 在现代 Viewport 2.0 下
不可靠，所以这里改成**输出真实网格**：

* ``outputBellMesh`` —— 变形后的钟形（C++ 版就有这个属性）
* ``outputRingMesh`` —— 全部碰撞环合并成的一个网格（**新增**）

把这两个接到 mesh shape 上就能看见、能选、能上材质，见 ``mesh_output.py``。
"""

import maya.OpenMaya as om
import maya.OpenMayaMPx as ompx

from utils import input_matrix
from bellColliderSolver import (BellColliderInputs, BellColliderOutputs,
                                BellColliderSolver)


NODE_NAME = "bellCollider"


class BellCollider(ompx.MPxLocatorNode):
    """``bellCollider`` 节点本体：DG 层的"薄壳"，只负责搬运数据。

    职责边界很清楚 —— 读属性 → 填 :class:`BellColliderInputs` → 调
    :meth:`BellColliderSolver.solve` → 把结果写回输出属性。**一行几何计算都不在
    这里**，全在 ``bellColliderSolver.py``。这样求解器可以脱离 Maya 节点单测
    （见 ``test_node.py``），节点这边改属性也不会碰到数学。

    **为什么继承 MPxLocatorNode 而不是 MPxNode**：locator 是 DAG 节点，
    在场景里有自己的 transform，可以被选中、被 K 动画、在大纲里看得见，
    也方便 ``scripts/colliders.py`` 把它当成一个可摆放的控制器。C++ 版还额外
    挂了 ``MPxDrawOverride`` 来画钟形，本版本**没有重写 ``draw()``**（原因见
    模块文档字符串），所以视口里看到的只是 Maya 默认的十字 locator 图标；
    真正的几何要靠 ``outputBellMesh`` / ``outputRingMesh`` 接到 mesh shape 上。

    属性一览（详细语义见 :class:`BellColliderInputs`）：

    * 输入 —— ``bellMatrix``（钟形世界矩阵）、``ringMatrix[]``（环世界矩阵数组）、
      ``bellSubdivision``、``ringSubdivision``、``bellBottomRadius``、
      ``falloff``、``collision``。
    * 输出 —— ``outputCurve``（底部轮廓 NURBS 曲线）、``outputBellMesh``
      （变形后的钟形网格）、``outputRingMesh``（全部环合并的显示网格，本版本新增）。

    **矩阵的语义约定**（全工程通用）：矩阵的平移是物体位置，**Y 轴（第 1 行）
    既是朝向又编码了长度/半径**，X/Z 轴长是横向缩放。所以不用单独的
    size / radius 属性 —— 直接缩放场景里那个 locator 就行。

    下面这一串 ``attr_*`` 是**类变量**（对应 C++ 的 static MObject 成员）：
    :func:`initialize` 在注册节点时给它们赋值，之后所有实例共享同一份属性
    句柄。所以 :meth:`compute` 里访问的都是 ``BellCollider.attr_xxx``
    而不是 ``self.attr_xxx``。
    """

    # 节点的唯一类型 ID，Maya 靠它在场景文件里认出这个节点。
    #
    # **注意这里和 C++ 版并不一致**：``sources/bellCollider.cpp`` 用的是十进制
    # 1274434，而这里是 0x01023（十进制 66083），落在 Autodesk 划给内部测试用的
    # 0x00000–0x7ffff 区段里。后果有两个：
    #   1. 用 C++ 版存的场景里那些 bellCollider 节点，本版本打开时认不出来；
    #   2. 反过来，本版本和 C++ 版的 bellCollider 类型 ID 不冲突，
    #      但**节点名**同为 "bellCollider"，Maya 仍会拒绝重复注册
    #      —— 所以 README 里"不要同时加载两个插件"的告诫依然成立。
    # （同目录的 planeCollider 用的才是与 C++ 一致的 1274435。）
    typeId = om.MTypeId(0x01023)

    attr_bellMatrix = om.MObject()
    attr_ringMatrix = om.MObject()
    attr_bellSubdivision = om.MObject()
    attr_ringSubdivision = om.MObject()
    attr_bellBottomRadius = om.MObject()
    attr_falloff = om.MObject()
    attr_collision = om.MObject()
    attr_outputCurve = om.MObject()
    attr_outputBellMesh = om.MObject()
    attr_outputRingMesh = om.MObject()

    def __init__(self):
        """由 :func:`creator` 在 Maya 每次新建该类型节点时调用，不要自己 new。

        除了把基类初始化一遍，这里**什么都不做** —— 节点不持有任何跨次求解的
        状态（缓存、上一帧结果之类）。这是有意为之：DG 节点必须是无状态的纯函数，
        否则时间轴来回拖、撤销重做、并行求值时结果会不一致。

        API 1.0 里基类构造必须显式写出来（``MPxLocatorNode.__init__(self)``），
        漏掉的话底层 C++ 对象没建起来，节点一被求值就崩。
        """
        ompx.MPxLocatorNode.__init__(self)

    def compute(self, plug, dataBlock):
        """Maya 在**下游需要**本节点某个输出、而该输出处于脏（dirty）状态时自动调用。

        绝不要手动调它。触发时机由 DG 决定：用户拖动钟形 locator → 该 locator 的
        worldMatrix 变化 → 顺着连接把 ``bellMatrix`` 标脏 → 由
        :func:`initialize` 里声明的 ``attributeAffects`` 把三个输出一并标脏 →
        视口要画 mesh 时才真正调到这里（**惰性求值**，改完属性不会立刻计算）。

        `plug` 是"当前要算哪一个输出"。Maya 一次只问一个插槽，但本节点三个输出
        共用同一套计算，所以**一次把三个都算了、也都 setClean**，
        后两次询问就会因为已经干净而被 DG 直接跳过。

        `dataBlock` 是本次求值的数据块。**输入只能通过它读、输出只能通过它写** ——
        直接去读上游节点的 plug 会绕过 DG 的求值上下文（不同时间/不同求值分支下
        拿到错的值）。本文件里唯一的例外见 ``utils.plug_vector`` 的说明，
        而 bellCollider 的属性都不是 3 分量型，所以这里全程走 dataBlock。

        返回值约定：不认识的插槽返回 ``om.kUnknownParameter``，等于告诉 Maya
        "这个我不管，你按默认逻辑处理"；认领了的插槽返回 None 即可。
        """
        # ---- 第 1 步：确认 Maya 问的是不是我们负责的输出 ----
        # 输入属性被标脏时 Maya 也可能拿别的插槽来问（比如内部的 message 属性），
        # 不属于这三个输出就必须原样交还，绝不能"顺手算一遍"。
        attr = plug.attribute()
        if attr not in (BellCollider.attr_outputCurve,
                        BellCollider.attr_outputBellMesh,
                        BellCollider.attr_outputRingMesh):
            return om.kUnknownParameter

        # ---- 第 2 步：读取输入，填进求解器的输入结构体 ----
        # 读取输入
        inputs = BellColliderInputs()
        # 必须用 utils.input_matrix 而不是直接 inputValue(...).asMatrix()：
        # asMatrix() 返回的是临时 handle 内部数据的**引用**，语句一结束就悬空，
        # 之后读出来是被复用的内存（症状：轴变零向量、坐标爆到十万量级，且不报错）。
        # 详见 utils.input_matrix 的说明。
        inputs.bellMatrix = input_matrix(dataBlock, BellCollider.attr_bellMatrix)

        # ringMatrix 是数组属性（一个节点可以接任意多个碰撞环）
        ringMatrixHandle = dataBlock.inputArrayValue(BellCollider.attr_ringMatrix)
        elementCount = ringMatrixHandle.elementCount()
        if elementCount == 0:
            # 提前退出，且**故意不 setClean** —— 输出插槽保持脏状态，
            # 下游读到的还是上一次的数据。等用户真的接上环再算。
            # 代价是 Maya 每次都会重新问一遍（脏值不会自己变干净），
            # 但这条分支几乎不进入计算，开销可以忽略。
            return  # 没有可碰撞的环，保持输出不变

        for i in range(elementCount):
            # 必须用 jumpToArrayElement（**物理**索引 0..elementCount-1），
            # 不能用 jumpToElement（那是**逻辑**索引）。用户在 AE 里删过某个
            # ringMatrix 元素后，逻辑索引会出现空洞（比如剩下 0、1、3），
            # 这时按逻辑索引遍历 0..2 会漏掉 3 号、还会在 2 号上报错。
            ringMatrixHandle.jumpToArrayElement(i)  # 按物理索引遍历
            # 同样必须拷贝：asMatrix() 返回的是 handle 内部引用
            inputs.ringMatrices.append(
                om.MMatrix(ringMatrixHandle.inputValue().asMatrix()))

        # 标量属性可以放心走 dataBlock —— asInt()/asFloat() 返回的是**值拷贝**，
        # 不像 asMatrix() 那样返回引用，所以不需要额外封装。
        # （3 分量属性则是另一回事，见 planeCollider.py 里 plug_vector 的用法。）
        inputs.bellSubdivision = dataBlock.inputValue(BellCollider.attr_bellSubdivision).asInt()
        inputs.ringSubdivision = dataBlock.inputValue(BellCollider.attr_ringSubdivision).asInt()
        inputs.bellBottomRadius = dataBlock.inputValue(BellCollider.attr_bellBottomRadius).asFloat()
        inputs.falloff = dataBlock.inputValue(BellCollider.attr_falloff).asFloat()
        inputs.collision = dataBlock.inputValue(BellCollider.attr_collision).asFloat()

        # ---- 第 3 步：调求解器 ----
        # 全部几何计算都在这一行里发生。solve 会把两个 MObject 数据对象填进
        # outputs（曲线数据 + 钟形网格数据），节点侧只管转交，不做二次加工。
        # 调用求解器
        outputs = BellColliderOutputs()
        BellColliderSolver.solve(inputs, outputs)

        # 环的显示网格与碰撞计算**无关**：solve 只用 ringMatrices 算变形，
        # 这里额外造一份可见几何纯粹是为了让用户看见环在哪。
        # 所以 ringSubdivision 只影响圆的光滑度，调它不会改变钟形的形状。
        # 参数 1 是"以 Y 轴为中心轴"，末尾的 1 是高度（半径用默认的 1，
        # 这个 1 正是 ringPush 反推环实际半径时的基准）。
        # 碰撞环合并成一个网格，供显示用
        ringMesh = BellColliderSolver.makeRingsMesh(
            inputs.ringMatrices, 1, inputs.ringSubdivision, 1)

        # ---- 第 4 步：写输出 ----
        # 曲线/网格这类复杂数据统一用 setMObject 交出数据对象本身，
        # 不需要（也不能）逐点拷贝。
        dataBlock.outputValue(BellCollider.attr_outputCurve).setMObject(outputs.outputCurveData)
        dataBlock.outputValue(BellCollider.attr_outputBellMesh).setMObject(outputs.outputBellMeshData)
        dataBlock.outputValue(BellCollider.attr_outputRingMesh).setMObject(ringMesh)

        # ---- 第 5 步：标记为干净 ----
        # setClean 是在告诉 DG "这个插槽算完了，值可以信"。**漏掉会有两种后果**：
        #   1. 插槽永远是脏的，Maya 每次求值都会重新调 compute —— 对这个
        #      纯 Python 的求解器来说是实打实的卡顿；
        #   2. 更糟的是可能触发反复求值甚至递归，Maya 会报
        #      "computing plug ... is not clean" 之类的告警。
        # 三个输出都要单独 setClean —— Maya 只问了其中一个插槽，但我们上面把三个
        # 都算并写了，一并标干净才能让后两次询问直接命中缓存、不再进 compute。
        dataBlock.setClean(BellCollider.attr_outputCurve)
        dataBlock.setClean(BellCollider.attr_outputBellMesh)
        dataBlock.setClean(BellCollider.attr_outputRingMesh)


def creator():
    """节点工厂：Maya 每次需要新建一个 ``bellCollider`` 实例时调用。

    在 :func:`colliders_Node.initializePlugin` 里作为
    ``MFnPlugin.registerNode`` 的第 3 个参数注册进去，之后由 Maya 内部调用
    （用户执行 ``cmds.createNode("bellCollider")`` 时就会走到这里）。

    ``ompx.asMPxPtr`` 是 API 1.0 的**所有权移交**：它把 Python 对象包成 Maya
    认识的裸指针，并把生命周期交给 Maya 管。少了这一层直接 ``return
    BellCollider()``，Python 这边引用一没就把对象回收了，而 Maya 还握着那块
    内存 —— 结果是随机崩溃。
    """
    return ompx.asMPxPtr(BellCollider())


def initialize():
    """建立节点的属性表和依赖关系。**整个 Maya 会话里只跑一次**。

    由 ``MFnPlugin.registerNode`` 在插件加载时调用（作为第 4 个参数注册），
    早于任何一个节点实例被创建。因此这里建出来的属性是**类级别**的：
    所有 ``bellCollider`` 实例共享同一批 MObject 句柄，存回
    ``BellCollider.attr_*`` 类变量里供 :meth:`BellCollider.compute` 使用。

    做三件事，顺序不能乱：

    1. ``create`` 出每个属性并 ``addAttribute`` 注册到节点上；
    2. 设置各自的 UI/存储标志（keyable、min/max、hidden、writable…）；
    3. 用 ``attributeAffects`` 声明"哪个输入变了要重算哪个输出"。

    **关于复用 function set**：这里 ``nAttr`` / ``mAttr`` / ``tAttr`` 各只建一个，
    然后反复 ``create``。这么写是安全的 —— 每次 ``create`` 都会把 function set
    重新绑定到新属性上，随后的 ``setMin`` / ``setKeyable`` 自然落在最新那个属性上。
    但 ``utils.create_numeric3`` 用的 ``createPoint`` **不能**这样复用（会让多个
    属性串在一起，改 A 动 B），所以那边坚持每次新建实例，两处写法不一致是有原因的。
    """
    nAttr = om.MFnNumericAttribute()
    mAttr = om.MFnMatrixAttribute()
    tAttr = om.MFnTypedAttribute()

    # bellMatrix：钟形的世界矩阵，通常从一个 locator 的 worldMatrix 连过来。
    # 平移 = 钟形底面圆心；Y 轴（第 1 行）**既是朝向、轴长又是钟形的高**；
    # X/Z 轴长 = 横向缩放。所以调整钟形的尺寸靠缩放那个 locator，
    # 节点上没有单独的 height / radius 属性。
    # 长名和短名故意写成一样的：矩阵属性一般由脚本连接、不手打，
    # 短名再短也没人用，写一样反而少一处要记的东西。
    BellCollider.attr_bellMatrix = mAttr.create("bellMatrix", "bellMatrix")
    mAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(BellCollider.attr_bellMatrix)

    # ringMatrix：碰撞环的世界矩阵**数组** —— 一个节点可以接任意多个环。
    # 每个元素的语义同 bellMatrix：Y 轴是环的朝向，X/Z 轴长是环的半径
    # （基础半径按 1 建模，见 BellColliderSolver.makeRingsMesh）。
    # setArray(True) 之后 compute 里必须用 inputArrayValue 而不是 inputValue。
    BellCollider.attr_ringMatrix = mAttr.create("ringMatrix", "ringMatrix")
    mAttr.setArray(True)
    mAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(BellCollider.attr_ringMatrix)

    # bellSubdivision：钟形一圈的分段数。它不只是精度参数 —— 求解器的顶点布局
    # 完全依赖它（0 是中心点，1..n 是底圈，n+1.. 是顶圈），
    # 所以 deformPoints 里那些 `range(bellSubdivision + 1, ...)` 才等价于"只动顶圈"。
    # 下限 3 是几何要求（少于 3 段围不成一个圈），上限 64 是防止 Python 求解器卡死。
    BellCollider.attr_bellSubdivision = nAttr.create(
        "bellSubdivision", "bellSubdivision", om.MFnNumericData.kInt, 16)
    nAttr.setMin(3)
    nAttr.setMax(64)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(BellCollider.attr_bellSubdivision)

    # ringSubdivision：**纯显示参数** —— 只决定 outputRingMesh 里每个环画得多圆，
    # 碰撞计算只读 ringMatrix，不读这个网格。所以调它绝不会改变钟形的变形结果。
    BellCollider.attr_ringSubdivision = nAttr.create(
        "ringSubdivision", "ringSubdivision", om.MFnNumericData.kInt, 16)
    nAttr.setMin(3)
    nAttr.setMax(64)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(BellCollider.attr_ringSubdivision)

    # bellBottomRadius：底圈相对顶圈的半径比（顶圈半径在局部空间恒为 1）。
    # < 1 是上宽下窄的裙摆，== 1 退化成圆柱。没有设上限，是因为做成上窄下宽的
    # 喇叭形也是合法需求。
    BellCollider.attr_bellBottomRadius = nAttr.create(
        "bellBottomRadius", "bellBottomRadius", om.MFnNumericData.kFloat, 0.8)
    nAttr.setMin(0)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(BellCollider.attr_bellBottomRadius)

    # falloff：变形的角度阈值，取值 -1..1，直接拿来和"顶点方向·环方向"的余弦比。
    # 默认 0 = 朝着环那半边（夹角小于 90°）的顶点都受影响；
    # 调大 → 受影响范围收窄；-1 = 整圈都受影响。
    BellCollider.attr_falloff = nAttr.create(
        "falloff", "falloff", om.MFnNumericData.kFloat, 0)
    nAttr.setMin(-1)
    nAttr.setMax(1)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(BellCollider.attr_falloff)

    # collision：径向挤压的混合强度，0 = 关闭（默认）。与 falloff 控制的倾斜变形
    # 是**两套独立效果**，叠加生效，见 bellColliderSolver.ringPush。
    # 注意环只是"相切"时没有任何效果，必须真的插进钟形里才看得到。
    BellCollider.attr_collision = nAttr.create(
        "collision", "collision", om.MFnNumericData.kFloat, 0)
    nAttr.setMin(0)
    nAttr.setMax(1)
    nAttr.setKeyable(True)
    ompx.MPxNode.addAttribute(BellCollider.attr_collision)

    # outputCurve：变形后钟形**顶圈**的轮廓曲线（C++ 版就有的属性），
    # 拿去驱动毛发/布料导引线或做 loft。
    # setHidden 只是不在 AE / 通道盒里露脸（曲线数据在 UI 里没法编辑，
    # 露出来只会碍事），脚本照样能 connectAttr 连它。
    BellCollider.attr_outputCurve = tAttr.create(
        "outputCurve", "outputCurve", om.MFnData.kNurbsCurve)
    tAttr.setHidden(True)
    ompx.MPxNode.addAttribute(BellCollider.attr_outputCurve)

    # 两个 mesh 输出是本 Python 版**替代 C++ MPxDrawOverride 的方案**：
    # 不在视口里画，而是吐出真实网格数据，用 mesh_output.py 接到 mesh shape 上，
    # 于是能选、能上材质、能渲染（详见 README）。
    #
    # 这两个标志是所有输出属性的标准配置，含义是：
    #   setWritable(False) —— 只出不进，禁止别人往里连或直接赋值；
    #   setStorable(False) —— 不写进 .ma/.mb 文件。网格数据每次都算得出来，
    #                          存进场景纯属浪费体积，还会在改代码后残留旧结果。
    # 输出网格对用户可见 —— 需要手动接到 mesh shape 上
    BellCollider.attr_outputBellMesh = tAttr.create(
        "outputBellMesh", "outputBellMesh", om.MFnData.kMesh)
    tAttr.setWritable(False)
    tAttr.setStorable(False)
    ompx.MPxNode.addAttribute(BellCollider.attr_outputBellMesh)

    # outputRingMesh：全部碰撞环合并成的**一个**网格（本 Python 版新增）。
    # 合并成一个而不是每环一个，是为了只用接一次线就能看到所有环。
    # 只是可视化，不参与任何计算。
    BellCollider.attr_outputRingMesh = tAttr.create(
        "outputRingMesh", "outputRingMesh", om.MFnData.kMesh)
    tAttr.setWritable(False)
    tAttr.setStorable(False)
    ompx.MPxNode.addAttribute(BellCollider.attr_outputRingMesh)

    # ---- 依赖声明：这是整个 initialize 里最容易出错、后果也最隐蔽的一段 ----
    #
    # attributeAffects(driver, output) 是在给 DG 画依赖图：driver 变脏时，
    # output 也跟着变脏，下次有人读 output 才会重新调 compute。
    #
    # **漏声明的后果**是脏值传不过去：改了属性但视口纹丝不动，用户以为节点坏了。
    # **多声明的后果**只是多算几次，功能正确。所以这里干脆取全部输入 × 全部输出
    # 的**笛卡尔积**，宁滥勿缺 —— 比如 bellSubdivision 其实和 outputRingMesh
    # 毫无关系（环的精度由 ringSubdivision 决定），照样连上，
    # 无非是改钟形分段时多重建一次环网格。
    drivers = (BellCollider.attr_bellMatrix, BellCollider.attr_ringMatrix,
               BellCollider.attr_bellSubdivision, BellCollider.attr_ringSubdivision,
               BellCollider.attr_bellBottomRadius, BellCollider.attr_falloff,
               BellCollider.attr_collision)

    outputs = (BellCollider.attr_outputCurve, BellCollider.attr_outputBellMesh,
               BellCollider.attr_outputRingMesh)

    for driver in drivers:
        for output in outputs:
            ompx.MPxNode.attributeAffects(driver, output)
