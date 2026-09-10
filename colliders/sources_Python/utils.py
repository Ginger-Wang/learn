"""``sources/utils.hpp`` 的 Python 移植版（Maya Python API 1.0）。

碰撞器节点共用的几何辅助函数，基于 ``maya.OpenMaya`` 编写。

API 1.0 的两个坑，本文件里都做了封装，其它模块直接用这里的函数即可：

* ``MMatrix`` 用 ``mat(row, col)`` 取元素，没有 ``getElement``；写元素要走
  ``MScriptUtil.setDoubleArray``。见 `maxis` / `set_maxis` / `matrix_from_rows`。
* ``float * MVector`` 不可用（没有 ``__rmul__``），一律写成 ``MVector * float``。
"""

import math

import maya.OpenMaya as om


# 角度 / 弧度换算常数，逐字照抄 utils.hpp 里的两个宏（不是用 math.pi 算出来的）。
# 之所以保留这两个精度不高的字面量，是为了跟 C++ 版算出完全一样的数：改成
# 180/pi 会让两边的结果在小数点后第五位开始分叉，对拍时很难判断是移植错误
# 还是精度差异。
#
# 注意它们**不是精确互为倒数**（57.2958 * 0.0174532862 ≈ 0.99999998），所以
# 不要拿它们做"度→弧度→度"的往返换算；真要高精度请直接用 math.degrees /
# math.radians。
#
# 目前 Python 移植版里没有任何地方用到它们（C++ 里也只是 utils.hpp 顺带定义的
# 公共宏），保留纯粹是为了与 C++ 源码一一对应。
RAD2DEG = 57.2958
DEG2RAD = 0.0174532862

# 改动本文件时顺手 +1。诊断脚本会打印它，用来确认 Maya 里跑的到底是不是磁盘上
# 的最新代码 —— 插件重载不彻底时，sys.modules 里可能还留着旧模块，表现就是
# "明明改了却没效果"。
CODE_REVISION = 7


class Plane(object):
    """由一个原点和一个单位法线定义的无限平面。

    是本工程里"把三维问题降到二维"的主力工具，出现在三个地方：

    * `planeCollider` —— 平面本身就是碰撞体，``distance() < 0`` 判断点是否穿到
      了背面，穿了就 ``projectPoint`` 拍回平面上。
    * `bellColliderSolver` —— 用钟形底面（原点 = 钟形平移，法线 = 钟形 Y 轴的
      单位向量）当参考平面，把环的位置/方向投影上去，这样"环从侧面顶钟形"就
      退化成平面内的二维问题；另外还用环的横截面（`ringPlane`）求环在某个
      方向上的实际半径。
    * `bellColliderMulti` / `test_node` —— 同上，多环版本各自建自己的平面。

    **法线方向是有语义的**：`distance` 返回的是有符号距离，法线指向的一侧为
    正。planeCollider 里那个 ``normalAxis`` 属性（0..5，>2 表示取负轴）就是让
    用户翻转这个符号，从而决定"往哪一侧推"。

    用 ``__slots__`` 是因为求解器的内层循环里会为每个环反复构造 Plane，省掉
    每个实例的 ``__dict__`` 能少一点分配开销。
    """

    __slots__ = ("orig", "normal")

    def __init__(self, orig=None, normal=None):
        """三种构造形态，对应 C++ 里的三个重载构造函数。

        * ``Plane()`` —— 空构造：原点为 (0,0,0)、法线为零向量。**此时法线不是
          单位向量**，直接拿去 `distance` / `projectPoint` 只会得到 0；空构造
          只用于"先占个位、之后整体赋值"的场景。
        * ``Plane(other)`` —— 拷贝构造：`orig` 传入的是另一个 Plane。这里对
          `orig` / `normal` 都**重新构造**了新的 MPoint / MVector，是刻意的深
          拷贝 —— API 1.0 里直接 ``self.orig = other.orig`` 会让两个 Plane 共享
          同一个对象，改一个动另一个。
        * ``Plane(origin, normal)`` —— 常规构造：`normal` 会被**自动归一化**，
          所以调用方可以放心传入未归一化的矩阵轴（例如
          ``maxis(planeMatrix, i)``，其长度代表缩放）。也因此，
          `Plane` 只保留方向、丢弃长度信息；需要长度的地方（如 planeCollider
          的圆盘半径）必须在构造之前另外取一次 ``.length()``。

        注意判断顺序：只有 ``orig`` 和 ``normal`` **都**为 None 才走空构造；
        写成 ``Plane(somePoint)`` 会掉进第三个分支，然后在
        ``MVector(None)`` 上报错。
        """
        if orig is None and normal is None:
            self.orig = om.MPoint()
            self.normal = om.MVector()
        elif isinstance(orig, Plane):
            self.orig = om.MPoint(orig.orig)
            self.normal = om.MVector(orig.normal)
        else:
            self.orig = om.MPoint(orig)
            self.normal = om.MVector(normal).normal()

    def projectVector(self, vec):
        """取 `vec` 位于平面内的分量（剔除法线方向的分量）。

        投影的是**方向**，与平面原点无关（只有 `projectPoint` 才用得到
        `orig`）。求解器用它把环的朝向压到钟形底面上，得到"环相对钟形的横向
        倾斜方向"；投影后长度会变短，长度趋于 0 就说明环轴与平面法线几乎平行
        （即环正对着钟形），此时方向无从定义，调用方会跳过倾斜变形。

        注意写法是 ``self.normal * (vec * self.normal)`` 而不是 C++ 里的
        ``(vec * normal) * normal`` —— API 1.0 的 MVector 没有 ``__rmul__``，
        ``float * MVector`` 会直接报错，只能把标量放右边。
        """
        return vec - self.normal * (vec * self.normal)

    def distance(self, src):
        """`src` 沿法线方向到平面的有符号距离。

        **符号才是重点**：正 = 在法线指向的一侧，负 = 在背面。planeCollider
        就是靠 ``distance(p) < 0`` 判断点有没有穿透平面，只有穿透的点才被拍
        回平面，没穿透的原样输出（单向碰撞，不会把外面的点吸过来）。

        `src` 先包一层 ``MPoint`` 是为了兼容调用方传 MVector 的情况；平面法线
        构造时已归一化，所以这里的点积直接就是距离，不需要再除以模长。
        """
        return (om.MPoint(src) - self.orig) * self.normal

    def projectPoint(self, src):
        """平面上距离 `src` 最近的点（沿法线的垂直投影）。

        对已经在平面正面的点也照投不误 —— 它不做"只投影穿透点"的判断，那是
        调用方的事（见 `distance`）。求解器用它把钟形顶点、环心统一压到钟形
        底面上，再在这个二维平面里做碰撞计算。
        """
        src = om.MPoint(src)
        dist = (src - self.orig) * self.normal
        return src - self.normal * dist

    def findLineIntersection(self, linePoint, lineDirection):
        """无限直线与平面的交点（直线与平面平行时结果未定义）。

        `lineDirection` 内部会被归一化，所以传未归一化的向量也没关系；直线是
        **无限长**的，交点可能落在 `linePoint` 的"后面"（d 为负），本函数不做
        任何区间裁剪。

        平行时分母 ``lineVector * self.normal`` 为 0：C++ 版会静默得到 inf /
        nan 一路传下去，**Python 版则直接抛 ZeroDivisionError**。这是两边行为
        不一致的地方 —— 调用方若可能喂进平行的直线，必须自己先用点积筛一遍。
        """
        lineVector = om.MVector(lineDirection).normal()

        d = ((self.orig - om.MPoint(linePoint)) * self.normal) / (lineVector * self.normal)
        return om.MPoint(linePoint) + lineVector * d


def clamp(v, l, h):
    """把 `v` 限制到 ``[l, h]`` 区间内，对应 C++ 里的 ``template<T> clamp``。

    行为上的几个约定，用之前先确认它们符合预期：

    * **不校验 l <= h**。传反了会直接返回 `h`（先判 ``v < l`` 命中就回 `l`，
      否则 ``v > h`` 命中回 `h`），不报错。
    * 判断只用 ``<`` / ``>``，所以 **NaN 会原样穿过**（NaN 与任何数比较都是
      False）。想靠它兜住除零产生的 NaN 是兜不住的。
    * 是泛型的：只要类型支持比较就行，MPoint/MVector 这类不支持 ``<`` 的类型
      不能传。

    C++ 版和当前的 Python 移植版里都**没有任何地方调用它**（C++ 的 utils.hpp
    只是把它当公共小工具定义在那儿）。保留是为了两边源码一一对应，别以为它在
    参与计算 —— 各节点里的权重/falloff 钳位都是就地用 min/max 写死的。
    """
    if v < l:
        return l
    if v > h:
        return h
    return v


def maxis(mat, index):
    """把 `mat` 的第 `index` 行取为向量（Maya 矩阵是行主序的）。

    API 1.0 的 MMatrix 用 ``mat(row, col)`` 取元素。

    行主序意味着：第 0/1/2 行分别是局部 X/Y/Z 轴在世界空间下的方向向量，
    第 3 行是平移。**这三个轴向量没有被归一化，它们的长度就是该方向上的
    缩放量** —— 这个性质在本工程里被到处利用（钟形的高、环的半径、
    planeCollider 圆盘的半径，全都是直接读轴长得来的），所以需要纯方向时
    务必自己 ``.normal()``，需要尺寸时才用 ``.length()``。

    第 4 列（w）被无条件丢弃：仿射变换里它对前三行恒为 0、对第 3 行恒为 1，
    对本工程用到的所有矩阵都成立。

    `index` 不做范围检查，越界会由 Maya 侧报错。
    """
    return om.MVector(mat(index, 0), mat(index, 1), mat(index, 2))


def xaxis(mat):
    """`mat` 的局部 X 轴（第 0 行）在世界空间中的方向，长度 = X 方向的缩放。

    `skirtBellCollider` 用它从腰部矩阵取出"身体正左/正右"的参考方向，再和
    `zaxis` 一起正交化出裙子的横截面坐标系。要纯方向请自行 ``.normal()``。
    """
    return maxis(mat, 0)


def yaxis(mat):
    """`mat` 的局部 Y 轴（第 1 行），长度 = Y 方向的缩放。

    本工程的核心约定：钟形和环的矩阵里，**Y 轴既是朝向、其长度又是高度**
    （钟形）或参与半径计算（环）。求解器里大量出现的
    ``maxis(matrix, 1)`` 就是这一行，含义与本函数完全相同。
    """
    return maxis(mat, 1)


def zaxis(mat):
    """`mat` 的局部 Z 轴（第 2 行），长度 = Z 方向的缩放。

    与 `xaxis` 配对使用：钟形/环的横向缩放由 X、Z 两轴的长度共同决定
    （各向同性时两者相等，即圆形截面；不等则是椭圆截面）。
    """
    return maxis(mat, 2)


def taxis(mat):
    """`mat` 的平移（第 3 行），包成 MPoint 返回。

    返回 MPoint 而不是 MVector 是有意为之：平移是**位置**不是方向，包成点
    之后才能直接参与 ``point - point → vector``、``point * matrix``
    （带平移的变换）这类运算；MVector 乘矩阵是会忽略平移分量的。

    在本工程里它就是各个碰撞体的"锚点"：钟形底面圆心、环心、髋/膝/踝关节
    的位置，都是这么取出来的。
    """
    return om.MPoint(maxis(mat, 3))


def set_maxis(mat, a, v):
    """写入 `mat` 的第 `a` 行。

    API 1.0 没有 setElement：``mat[a]`` 拿到的是该行的 double 指针，
    再用 MScriptUtil.setDoubleArray 逐个写。

    是**原地修改**：`mat` 本身被改掉，返回值只是为了方便链式书写，不是副本。
    只写该行的前三列，第 4 列（w）保持原样 —— 所以对第 3 行用它设置平移是
    安全的（w 仍是 1），不会把矩阵变成非仿射。
    """
    row = mat[a]
    om.MScriptUtil.setDoubleArray(row, 0, v.x)
    om.MScriptUtil.setDoubleArray(row, 1, v.y)
    om.MScriptUtil.setDoubleArray(row, 2, v.z)
    return mat


def mscale(mat):
    """取出 `mat` 的三轴缩放：把 X/Y/Z 三行的**长度**装进一个 MVector。

    这正是 `maxis` 那条约定的直接应用 —— 轴长即缩放。对本工程的矩阵来说，
    结果读起来就是 ``(横向半径, 高度, 横向半径)``。

    几点限制，别当成通用的矩阵分解：

    * 长度恒为非负，**取不出负缩放**（镜像矩阵会被读成正的）。
    * 有切变（shear）时结果没有意义 —— 它只量长度，不管三轴是否正交。
    * 与 ``MTransformationMatrix.getScale()`` 不同，这里不涉及任何坐标系
      参数，纯粹是行长度。

    C++ 版和当前 Python 移植版里都还没有调用点，属于对照 utils.hpp 保留的
    公共工具。
    """
    return om.MVector(maxis(mat, 0).length(),
                      maxis(mat, 1).length(),
                      maxis(mat, 2).length())


def set_mscale(mat, scale):
    """把 `mat` 的三轴缩放**设成** `scale`（`mscale` 的逆操作）。

    做法是"归一化再乘"：保留每个轴原有的**方向**，只替换长度。因此旋转和
    平移完全不受影响，是在本工程的约定下直接改"钟形高度 / 环半径"的正确
    手段 —— 比重新拼一个矩阵安全得多。

    * **原地修改 `mat`，返回 None**，不要写成 ``mat = set_mscale(mat, s)``。
    * `scale` 需要是有 ``.x/.y/.z`` 的对象（MVector），传三元组会报错。
    * 退化轴（长度为 0）上 ``normal()`` 返回零向量，乘完还是零向量：该轴会
      **静默地留在零**，设不上新长度也不报错。矩阵可能来源于缩放为 0 的
      变换时要留意。

    同 `mscale`，目前两边都还没有调用点。
    """
    set_maxis(mat, 0, maxis(mat, 0).normal() * scale.x)
    set_maxis(mat, 1, maxis(mat, 1).normal() * scale.y)
    set_maxis(mat, 2, maxis(mat, 2).normal() * scale.z)


def findSphereLineIntersection(linePoint, lineDirection, sphereCenter, sphereRadius):
    """无限直线与球面的两个交点。

    当直线与球面相离或相切时返回空数组。

    这是整个钟形碰撞的计算内核，`bellColliderSolver.deformPoints` 用它两次：
    先在钟形**局部空间**里求"环轴射线打在钟形侧壁上的哪一点"（局部空间下
    钟形顶圈半径恰好是 1，所以问题变成与单位球求交），再在世界空间里求
    "该点应该被推到的目标位置"。

    参数约定：`lineDirection` 内部会归一化，随便传；直线是**无限长**的，
    交点可能在 `linePoint` 的反方向上（d 为负），本函数不裁剪 —— 求解器是
    自己拿 ``(交点 - 环心) * 环轴 > 0`` 来挑正确那一侧的。

    返回的两点顺序固定：``[0]`` 是 ``+sqrt(delta)`` 那个根（沿直线方向更远
    的一侧），``[1]`` 是 ``-sqrt`` 那个根。调用方按下标取，别改顺序。

    **相切（delta == 0，恰好一个交点）被当成"没打中"处理**，和相离一样返回
    空数组。这是刻意的：相切意味着挤压量为零，硬算出来的那个点会让后续的
    方向计算退化成零向量；与其传一个不稳定的解出去，不如让调用方走"无碰撞"
    分支。求解器那边为此还特意把球半径放大到 1.001 留了余量，就是为了避免
    正好卡在相切上。
    """
    # 直线方程 P(d) = linePoint + d * lineVector，代入球面方程展开成
    # d² + b*d + c = 0。因为 lineVector 已归一化，二次项系数 a 恒等于 1，
    # 所以下面的求根公式里没有除以 2a、判别式里也是 b² - 4c 而不是 b² - 4ac。
    lineVector = om.MVector(lineDirection).normal()

    # b = 2 * (方向 · (起点 - 球心))
    b = 2 * (lineVector.x * (linePoint.x - sphereCenter.x) +
             lineVector.y * (linePoint.y - sphereCenter.y) +
             lineVector.z * (linePoint.z - sphereCenter.z))
    # c = |起点 - 球心|² - 半径²（逐分量手写而不是用 MVector.length()，
    # 与 C++ 版保持完全一致的浮点运算顺序，方便两边对拍）
    c = (math.pow(linePoint.x - sphereCenter.x, 2) +
         math.pow(linePoint.y - sphereCenter.y, 2) +
         math.pow(linePoint.z - sphereCenter.z, 2) -
         math.pow(sphereRadius, 2))

    delta = math.pow(b, 2) - 4 * c
    if delta <= 0:  # 交点为 0 个或 1 个时
        return om.MPointArray()

    # 两个根 = 沿归一化方向走的有符号距离（因为方向是单位长度，d 本身就是
    # 到 linePoint 的距离，符号表示在方向的正侧还是反侧）
    d1 = (-b + math.sqrt(delta)) / 2.0
    d2 = (-b - math.sqrt(delta)) / 2.0

    # 把两个参数值代回直线方程得到交点。逐分量写是因为 API 1.0 里
    # ``MPoint + MVector * float`` 这种链式写法不如手写可靠（且与 C++ 一致）。
    p1 = om.MPoint(linePoint.x + lineVector.x * d1,
                   linePoint.y + lineVector.y * d1,
                   linePoint.z + lineVector.z * d1)

    p2 = om.MPoint(linePoint.x + lineVector.x * d2,
                   linePoint.y + lineVector.y * d2,
                   linePoint.z + lineVector.z * d2)

    points = om.MPointArray()
    points.append(p1)
    points.append(p2)
    return points


# ---------------------------------------------------------------------------
# 以下辅助函数在 C++ 中没有对应实现：它们替代的是 C++ 里的隐式类型转换或
# 运算符，而 API 1.0 的 Python 绑定不提供这些。
# ---------------------------------------------------------------------------

def matrix_from_rows(rows):
    """从 4 行（每行 4 个数）构造 MMatrix。

    API 1.0 不能直接用列表构造 MMatrix，必须过一遍 MScriptUtil。

    这里带了自检和回退：``createMatrixFromList`` 在部分 Maya 版本上会**静默
    失败**（不报错，但矩阵还是单位矩阵）。那种情况下用它拼出来的钟形矩阵三个
    轴全变成世界轴，曲面会直接退化成一个平面 —— 而且不会有任何报错，极难查。
    所以写完校验两个元素，不对就改用逐元素写入（``mat[row]`` 拿行指针 +
    ``setDoubleArray``）。两条路都不通就抛异常，绝不返回一个悄悄错掉的矩阵。
    """
    values = []
    for row in rows:
        values.extend(row)

    mat = om.MMatrix()

    try:
        om.MScriptUtil.createMatrixFromList(values, mat)
        if _matrix_matches(mat, values):
            return mat
    except Exception:
        pass

    # 回退：逐元素写。mat[r] 在 API 1.0 里返回该行的 double 指针。
    mat = om.MMatrix()
    try:
        for r in range(4):
            rowPtr = mat[r]
            for c in range(4):
                om.MScriptUtil.setDoubleArray(rowPtr, c, values[r * 4 + c])
    except Exception as exc:
        raise RuntimeError(
            "无法构造 MMatrix：createMatrixFromList 和 setDoubleArray 都不可用"
            "（%s: %s）" % (type(exc).__name__, exc))

    if not _matrix_matches(mat, values):
        raise RuntimeError(
            "无法构造 MMatrix：写入后校验不通过，createMatrixFromList 和 "
            "setDoubleArray 都没有真正写进去")

    return mat


def _matrix_matches(mat, values):
    """校验矩阵内容是否真的被写进去了。

    只查四个角上的元素：足够区分"写成功"和"还是单位矩阵"这两种情况。
    """
    for r, c in ((0, 0), (1, 1), (2, 2), (3, 0)):
        if abs(mat(r, c) - values[r * 4 + c]) > 1e-9:
            return False
    return True


def create_numeric3(name, default=None, minValue=None, maxValue=None,
                    keyable=False):
    """创建 3 分量 double 属性，等价于 C++ 里的 ``MFnNumericData::k3Double``。

    API 1.0 里不能写 ``create(name, name, k3Double)`` 再 ``setDefault(x, y, z)``
    —— 那样 setDefault 会抛 kInvalidParameter("The data is not of that type")。
    必须用 ``createPoint()``，它直接建出 3 个 double 子属性（X/Y/Z），
    子属性命名和 k3Double 一致，所以读写代码不用改。

    **每个属性必须用独立的 MFnNumericAttribute 实例**，所以这里自己新建一个、
    不接受外部传入。多个属性共用同一个 function set 反复 ``createPoint`` 会让
    它们串在一起：表现是在属性编辑器里改 ``bellScale`` 却动了 ``ringScale``，
    而且 ``setDefault`` 也会落到错误的属性上（默认值看着就是不生效）。

    **注意 createPoint 建出来的子属性是 float，不是 double。**（实测：
    ``MFnNumericAttribute(child).unitType()`` 返回 ``kFloat``。）所以读这类属性
    **必须走 :func:`plug_vector`**，不能用 ``dataBlock.inputValue(attr)`` 那条路
    —— 它会用 ``asDouble()`` 把 4 字节当 8 字节读，读进相邻属性的内存，表现是
    "改一个属性动了另一个"。

    min / max 只是通道盒里的 UI 限制，个别 Maya 版本上三参数形式可能不被接受，
    所以这里容错处理：设不上就跳过，不影响功能。
    """
    nAttr = om.MFnNumericAttribute()
    attr = nAttr.createPoint(name, name)

    if default is not None:
        nAttr.setDefault(default[0], default[1], default[2])

    for label, values, setter in (("最小值", minValue, nAttr.setMin),
                                  ("最大值", maxValue, nAttr.setMax)):
        if values is None:
            continue
        try:
            setter(values[0], values[1], values[2])
        except Exception as exc:
            om.MGlobal.displayWarning(
                "属性 %s 的%s设置失败（仅影响通道盒限制）：%s" % (name, label, exc))

    if keyable:
        nAttr.setKeyable(True)

    _verify_numeric3(attr, name)
    return attr


def _verify_numeric3(attr, name):
    """确认建出来的属性名字对、而且真有 3 个子属性。

    属性串味是极难查的：节点能正常加载、AE 里也能看到控件，只是读到的值不对。
    在创建的当口就核对一遍，有问题直接抛出来，比等到几何形状出错再回头查
    省事得多。
    """
    try:
        created = om.MFnAttribute(attr).name()
    except Exception as exc:
        raise RuntimeError("属性 %s 创建后无法读取名字：%s: %s"
                           % (name, type(exc).__name__, exc))

    if created != name:
        raise RuntimeError(
            "属性名不符：期望 %r，实际建出来的是 %r —— "
            "说明 function set 被串用了" % (name, created))

    try:
        childCount = om.MFnCompoundAttribute(attr).numChildren()
    except Exception as exc:
        raise RuntimeError("属性 %s 不是 compound：%s: %s"
                           % (name, type(exc).__name__, exc))

    if childCount != 3:
        raise RuntimeError("属性 %s 应该有 3 个子属性，实际 %d 个"
                           % (name, childCount))


def copy_points(points):
    """MPointArray 赋值是引用绑定，所以需要显式拷贝。

    这是求解器里必须记住的一条纪律：``a = b`` 之后两个名字指向**同一份**点
    数组，改 a 就等于改 b。钟形网格的"基准点"（未变形的原始形状）只读一次，
    每个环、每一轮迭代都必须先 ``copy_points`` 出一份再改，否则上一个环的
    变形结果会被当成下一个环的输入，多环叠加时形状会越滚越离谱。
    """
    return om.MPointArray(points)


def lerp_point(a, b, t):
    """等价于 `a * (1 - t) + b * t`，避免依赖 MPoint 之间的加法。

    存在的理由有两个：API 1.0 里 ``MPoint + MPoint`` 不可用（点加点没有几何
    意义），而且 ``float * MPoint`` 也不可用。所以只能逐分量手写。

    `t` **不做钳位**：调用方（`deformPoints` 的权重、`skirtBellCollider` 的
    tightness）保证它已经在 0..1 内；传出界的值会得到外插结果而不是报错。

    用 ``a*(1-t) + b*t`` 而不是 ``a + (b-a)*t``，是为了在 t=0 / t=1 时能得到
    **精确**的端点值（后者会引入一次减法带来的舍入误差）—— 变形权重为 0 的
    点必须与原始点逐位相同，否则网格上会出现肉眼可见的细微抖动。

    w 分量不参与插值，结果是默认的 1。
    """
    it = 1.0 - t
    return om.MPoint(a.x * it + b.x * t,
                     a.y * it + b.y * t,
                     a.z * it + b.z * t)


def midpoint(a, b):
    """两点的中点。等价于 ``lerp_point(a, b, 0.5)``，但语义更直白。

    同样是为了绕开 API 1.0 —— ``(a + b) * 0.5`` 里的点加点是不允许的。

    `skirtBellCollider` 用它把左右髋关节的位置合成一个"骨盆中心"
    （``layout.H = midpoint(LH, RH)``），作为裙子各层沿轴测距的起点。取中点
    而不是取某一侧，是为了让角色单腿抬起时裙子的参考轴不会跟着摆动。

    只算 x/y/z，**不做齐次除法**：输入若是 w != 1 的非齐次点，结果不是几何
    意义上的中点。本工程里所有点的 w 都是 1，不用担心。
    """
    return om.MPoint((a.x + b.x) * 0.5, (a.y + b.y) * 0.5, (a.z + b.z) * 0.5)


def scaled_color(color, factor):
    """MColor * float —— 与 C++ 一致：只缩放 rgb，保持 alpha 不变。

    补上这个函数是因为 API 1.0 的 MColor 没有和标量相乘的运算符，而 C++ 的
    绘制代码里到处是 ``drawData.color * 0.5``（bellCollider.cpp：钟形用原色
    画、各个环用半暗的同色画，好一眼区分主体和碰撞体）。

    **alpha 必须原样保留**，这是关键点：不透明度是用户在节点上单独设的
    ``drawOpacity``，如果跟着 rgb 一起乘，环就会既变暗又变透明，视口里几乎
    看不见 —— 变暗和变透明是两件事。

    不做 0..1 钳位：factor > 1 会得到超出范围的颜色值，交给 Maya 的绘制端
    自己处理。

    目前 Python 移植版还没有实现视口绘制覆盖（draw override），所以这个函数
    暂时没有调用点，是为了对齐 C++ 而提前备好的。
    """
    return om.MColor(color.r * factor, color.g * factor, color.b * factor, color.a)


def ramp_value_at(rampAttr, position):
    """查询渐变曲线在 `position` 处的值。

    API 1.0 的 ``getValueAtPosition`` 用 out-param 返回，必须借 MScriptUtil
    分配一个 float 指针。

    `position` 是 0..1 的归一化参数；`skirtBellCollider` 用它查
    ``bellScaleRamp``，把"裙子从腰到摆的粗细变化"这条用户可编辑的曲线采样成
    每一层的缩放系数。

    两个容易踩的点：

    * **``util`` 必须留在局部变量里**（不能写成一行链式调用）。它持有那块
      内存，一旦被回收，``valuePtr`` 就是野指针，读出来是随机数还不报错。
    * 先 ``createFromDouble`` 再 ``asFloatPtr``：故意用更宽的 double 开缓冲，
      再按 float 读回 —— 反过来（createFromFloat + asDoublePtr）会越界读。
      ramp 的值本身是 float，所以最后用 ``getFloat`` 取。

    `position` 强转 float 是因为 API 对参数类型很挑，传进来的可能是 int 或
    numpy 标量。
    """
    util = om.MScriptUtil()
    util.createFromDouble(0.0)
    valuePtr = util.asFloatPtr()
    rampAttr.getValueAtPosition(float(position), valuePtr)
    return om.MScriptUtil.getFloat(valuePtr)


def input_matrix(dataBlock, attr):
    """从 dataBlock 取矩阵，并**立即拷贝一份**。

    这是 API 1.0 最阴的一个坑：``MDataHandle.asMatrix()`` 返回的是 handle
    内部数据的**引用**，不是拷贝。而 ``dataBlock.inputValue(attr)`` 产生的是
    临时 handle，这条语句一结束就被回收，引用随之悬空 —— 之后读出来的是已经
    被复用的内存，值看着像随机数。

    症状很难认：矩阵读出来时对时错、算出来的轴变成零向量、坐标爆到十万量级，
    而且不会有任何报错。所以一律走这个封装，用 MMatrix 的拷贝构造存下来。
    """
    return om.MMatrix(dataBlock.inputValue(attr).asMatrix())


def plug_vector(nodeObj, attr):
    """用 MPlug 读 3 分量属性。**这是唯一验证过能读对的路径。**

    为什么不走 dataBlock：``createPoint`` 建出来的属性上，
    ``dataBlock.inputValue(attr).child(...).asDouble()`` 读出来是垃圾值 ——
    子属性的实际数据类型和 ``asDouble()`` 期望的字长不一致，按 double 去解释
    4 字节的数据，结果毫无意义。实测长这样::

        MPlug   读到  ringScale=(4.455, 1.000,   4.455)
        dataBlock 读到 ringScale=(0.008, 977.921, 250347.657)

    而且相邻两个属性会读到重叠的内存（``ringScale.z`` 约等于
    ``bellScale.y``），表现就是"改一个属性动了另一个"。

    代价：绕过 dataBlock 意味着不参与 DG 的脏值优化。对这几个标量属性来说
    可以忽略 —— 正确性优先。
    """
    plug = om.MPlug(nodeObj, attr)
    if plug.numChildren() < 3:
        return om.MVector()
    return om.MVector(plug.child(0).asDouble(),
                      plug.child(1).asDouble(),
                      plug.child(2).asDouble())


def matrix_from_data(dataObj):
    """从矩阵数据对象取 MMatrix 并拷贝。

    ``MFnMatrixData(obj).matrix()`` 同样返回内部引用，而那个临时的
    MFnMatrixData 出了语句就没了。
    """
    if dataObj.isNull():
        return om.MMatrix()
    return om.MMatrix(om.MFnMatrixData(dataObj).matrix())


def mesh_points(meshFn):
    """MFnMesh.getPoints() 的 out-param 版本封装。

    API 1.0 的 getter 普遍是 C++ 风格的"传一个空容器进去让它填"，直接调用
    很啰嗦，这里统一封成返回值形式。

    默认取的是**物体空间**（getPoints 不传 space 参数时的默认值），求解器
    正是要这个：钟形网格的顶点在自己的局部空间里，再由 bellMatrix 变换到
    世界空间。

    返回的数组是新建的，与网格数据无关，可以放心修改（不像
    :func:`input_matrix` 那种引用悬空的场合）；但仍要注意后续在 Python 里把
    它赋给别的名字时是引用绑定，见 :func:`copy_points`。
    """
    points = om.MPointArray()
    meshFn.getPoints(points)
    return points


def mesh_triangles(meshFn):
    """MFnMesh.getTriangles() 的 out-param 版本封装，返回 (counts, indices)。

    `counts[i]` 是第 i 个多边形被三角化后的三角形个数，`indices` 是**扁平**
    的顶点下标流（每 3 个一组），两者的长度对不上是正常的。

    对应 C++ bellCollider.cpp 里 ``drawMesh`` 的取法：视口绘制需要把网格拆成
    三角形喂给 MUIDrawManager。Python 版尚未实现绘制覆盖，所以这个封装目前
    没有调用点。
    """
    counts = om.MIntArray()
    indices = om.MIntArray()
    meshFn.getTriangles(counts, indices)
    return counts, indices


def surface_cvs(surfaceFn, space=None):
    """MFnNurbsSurface.getCVs() 的 out-param 版本封装。

    `space` 默认取 ``MSpace.kObject``，而**不是在函数签名里写默认值** ——
    默认参数在模块导入时求值，而 ``om.MSpace.kObject`` 要求 OpenMaya 已经
    完成初始化；写进签名会让这个模块在某些加载时机（比如 Maya 尚未起完
    就被 import）直接失败。所以用 None 当哨兵、进函数体再取。

    返回的 CV 是按 U、V 展平成一维的 MPointArray（顺序为 U 主序），需要行列
    结构的话得自己按 ``numCVsInV`` 切分。

    `skirtBellCollider` 生成裙子的 NURBS 曲面用的是 ``MFnNurbsSurface.create``
    这条路，并不回读 CV，所以这个封装目前没有调用点，属于备用工具。
    """
    if space is None:
        space = om.MSpace.kObject
    cvs = om.MPointArray()
    surfaceFn.getCVs(cvs, space)
    return cvs
