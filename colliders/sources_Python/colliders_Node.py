"""``sources/main.cpp`` 的 Python 移植版 —— 插件入口（Maya Python API 1.0）。

在 Maya 中这样加载::

    import maya.cmds as cmds
    cmds.loadPlugin(r"<path_to_colliders>/sources_Python/colliders_Node.py")

注册的三个节点与 C++ 插件同名：``bellCollider``、``planeCollider``
和 ``skirtBellCollider``。

与 C++ 版的区别：不注册 drawDbClassification / drawOverride，因为本版本
不做视口绘制 —— 几何以网格属性输出，用 ``mesh_output.py`` 接到真实 shape 上。
"""

import os
import sys

import maya.OpenMaya as om
import maya.OpenMayaMPx as ompx


# 插件管理器里显示的 Vendor 和 Plug-in version
AUTHOR = "test--nodes"
VERSION = "1.0.0"
REQUIRED_API_VERSION = "Any"


# 本插件依赖的同级模块。卸载时会从 sys.modules 里清掉，
# 这样改完代码重新加载插件就能生效，不必重启 Maya。
_SIBLING_MODULES = ("utils", "bellColliderSolver", "bellCollider",
                    "bellColliderMulti", "planeCollider", "skirtBellCollider")

# 已成功注册的 (节点名, 类型 ID)，供 uninitializePlugin 使用
_registered = []


def _normalize_dir(path):
    """把可能是文件路径也可能是目录路径的字符串统一成目录。

    存在的原因：:func:`_plugin_dir` 那三条兜底路径返回的东西**形态不一致** ——
    ``__file__`` 和 ``cmds.pluginInfo(path=True)`` 给的是 .py 文件的完整路径，
    而 ``MFnPlugin.loadPath()`` 给的已经是目录了。要往 ``sys.path`` 里塞的必须是
    目录，所以在这里统一收口，三条路径各自判断一遍太啰嗦也容易漏。

    先 ``abspath`` 是因为 Maya 有时会给出相对路径或带 ``..`` 的路径，
    那种路径塞进 sys.path 会随当前工作目录变化而失效。

    用 ``isdir`` 实地探测而不是看有没有扩展名 —— 目录名里带点的情况真实存在。
    """
    path = os.path.abspath(path)
    return path if os.path.isdir(path) else os.path.dirname(path)


def _plugin_dir(pluginFn):
    """定位本文件所在目录。

    Maya 执行 .py 插件时不保证会定义 ``__file__``（Maya 2025 就不定义），
    所以依次用三种方式兜底。

    收的是**已经构造好的** MFnPlugin，不是 MObject —— 这里绝不能自己
    ``MFnPlugin(plugin)`` 一下：那样用的是默认参数，会把插件的
    Vendor / Plug-in version 当场登记成 "Unknown"，之后再带参数构造也覆盖
    不回来，插件管理器里就只剩 Unknown 了。
    """
    path = globals().get("__file__")
    if path:
        return _normalize_dir(path)

    try:
        path = pluginFn.loadPath()
        if path:
            return _normalize_dir(path)
    except Exception:
        pass

    try:
        import maya.cmds as cmds
        # 用插件自己的注册名，这样本文件改名也不影响
        path = cmds.pluginInfo(pluginFn.name(), query=True, path=True)
        if path:
            return _normalize_dir(path)
    except Exception:
        pass

    # 三条路都不通就直接抛，不要"猜"一个目录 —— 猜错的话后面 import 会失败在
    # 更奇怪的地方（比如 sys.path 里恰好有另一个同名 utils 模块），更难查。
    raise RuntimeError(
        "无法定位 sources_Python 目录，请先手动把它加入 sys.path 再加载插件")


def initializePlugin(plugin):
    """插件入口。**由 Maya 在 ``cmds.loadPlugin(...)`` 时自动调用**，不要手动调。

    函数名和签名是 Maya 规定的固定契约：Maya 执行这个 .py 文件后，会在模块的
    全局命名空间里找名叫 ``initializePlugin`` 的函数并传入一个 MObject。
    改名字插件就加载不了。

    做四件事：

    1. 构造 MFnPlugin，登记 Vendor / 版本号；
    2. 定位自身目录并塞进 ``sys.path``（否则同级模块 import 不到）；
    3. import 同级模块（**必须在这之后**，所以 import 只能写在函数体里）；
    4. 逐个 ``registerNode``，并把成功的记进 :data:`_registered`。

    任何一步失败都会 ``raise``，Maya 会把插件标记为加载失败。这是刻意的：
    半加载状态（一部分节点注册了、一部分没有）比干脆加载不上更难排查。
    """
    # 必须最先构造，且只构造这一次 —— Vendor / 版本号就是在这里登记的。
    # 后面 _plugin_dir 收的是这个**已构造好的** pluginFn 而不是原始的 plugin
    # MObject，就是为了防止那边再 MFnPlugin(plugin) 一下：那样用默认参数构造，
    # 会把 Vendor / 版本当场改成 "Unknown"，而且之后再带参数构造也覆盖不回来。
    pluginFn = ompx.MFnPlugin(plugin, AUTHOR, VERSION, REQUIRED_API_VERSION)

    # 同级模块只是本文件旁边的普通 .py 文件，必须先把本目录加入 sys.path，
    # 因此这些 import 只能放在函数内部，不能放到模块顶层。
    pluginDir = _plugin_dir(pluginFn)
    if pluginDir not in sys.path:
        sys.path.insert(0, pluginDir)

    import bellCollider
    import bellColliderMulti
    import skirtBellCollider

    # 四元组 —— registerNode 需要的全部信息，逐项含义：
    #   nodeName   在 MEL/Python 里 createNode 用的名字，也是 Maya 判重的依据
    #              （所以和 C++ 版同名的节点无法共存，见 README）；
    #   typeId     写进场景文件、用来认出节点类型的唯一 ID；
    #   creator    每次新建实例时的工厂函数；
    #   initialize 建属性表的函数，整个会话只被调一次。
    #
    # **注意 planeCollider 不在这个列表里** —— 模块文档字符串写的是"注册三个节点：
    # bellCollider、planeCollider、skirtBellCollider"，但实际注册的第二项是本
    # Python 版新增的 bellColliderMulti。``planeCollider.py`` 写得好好的，
    # ``_SIBLING_MODULES`` 里也留着它，就是没被 import 和 register，
    # 所以加载插件后 ``createNode("planeCollider")`` 会报"未知节点类型"。
    # 需要用它的话，在下面补一项 (planeCollider.NODE_NAME,
    # planeCollider.PlaneCollider.typeId, planeCollider.creator,
    # planeCollider.initialize) 并在上面加一行 import 即可。
    # （节点名, 类型 ID, creator, initialize）
    nodes = (
        (bellCollider.NODE_NAME, bellCollider.BellCollider.typeId,
         bellCollider.creator, bellCollider.initialize),
        (bellColliderMulti.NODE_NAME, bellColliderMulti.BellColliderMulti.typeId,
         bellColliderMulti.creator, bellColliderMulti.initialize),
        (skirtBellCollider.NODE_NAME, skirtBellCollider.SkirtBellCollider.typeId,
         skirtBellCollider.creator, skirtBellCollider.initialize),
    )

    # 清空历史记录再开始。**必须写成 `del _registered[:]`（原地清空）而不是
    # `_registered = []`** —— 后者在函数里是给局部变量赋值，模块级的那个列表
    # 纹丝不动，于是 unload/load 几次之后 _registered 里会堆满重复项，
    # uninitializePlugin 反复 deregister 同一个 typeId 而报错。
    del _registered[:]

    for nodeName, typeId, creator, initialize in nodes:
        # try:
        #     pluginFn.registerNode(nodeName, typeId, creator, initialize)
        # except Exception:
        #     om.MGlobal.displayError("节点注册失败：%s" % nodeName)
        #     raise
        try:
            pluginFn.registerNode(nodeName, typeId, creator, initialize)
            print("%s initialized..."%(nodeName))
        except Exception:
            # 这里用 sys.stderr.write 而不是 MGlobal.displayError（上面注释掉的
            # 那版）：插件加载失败时 Maya 的脚本编辑器/UI 未必已经就绪，
            # 写 stderr 至少能进控制台日志，不会连报错都看不到。
            # 写完立刻 raise，让 Maya 把整个插件判为加载失败 —— 只注册了一半的
            # 插件比彻底加载不上更难排查（节点存在但属性缺失，场景会静默出错）。
            # om.MGlobal.displayError("节点注册失败：%s" % nodeName)
            sys.stderr.write("Failed to register node: %s" % nodeName)
            raise

        # 只记**已经成功注册**的，这样即便中途失败，uninitializePlugin
        # 也不会去 deregister 一个根本没注册上的 typeId。
        _registered.append((nodeName, typeId))


def uninitializePlugin(plugin):
    """插件出口。**由 Maya 在 ``cmds.unloadPlugin(...)`` 时自动调用**，不要手动调。

    和 :func:`initializePlugin` 一样是 Maya 规定的固定函数名。职责是把
    initializePlugin 干过的事逐项撤销，顺序相反：

    1. 逐个 ``deregisterNode``（只注销 :data:`_registered` 里记着的，
       即真正注册成功过的）；
    2. 清空 :data:`_registered`；
    3. 把同级模块从 ``sys.modules`` 里踢掉。

    第 3 步是这个工程日常开发的关键 —— 详见下面的行内注释。

    ``sys.path`` 里插入的那个目录**故意不撤销**：撤了也没好处，而且别的东西
    可能已经依赖它了。重复 load 时 initializePlugin 里的 ``not in`` 判断
    会防止重复插入。

    注意 Maya 在还有该类型节点存在于场景里时会拒绝卸载插件，所以这里不必
    自己去清理场景中的节点实例。
    """
    # 同样带上参数构造，保持和 initializePlugin 一致
    pluginFn = ompx.MFnPlugin(plugin, AUTHOR, VERSION, REQUIRED_API_VERSION)

    for nodeName, typeId in _registered:
        try:
            # deregisterNode 只认 typeId，不认名字。元组里那个 nodeName
            # 留着纯粹是为了出错时能报出人看得懂的名字。
            pluginFn.deregisterNode(typeId)
        except Exception:
            # 卸载阶段 UI 是活的，可以放心用 displayError（和注册那边不同）
            om.MGlobal.displayError("节点注销失败：%s" % nodeName)
            raise

    del _registered[:]

    # 这一步是本工程能"改完代码 unload/load 一次就生效"的**全部原因**。
    # Python 的 import 有缓存：模块一旦进了 sys.modules，后续 import 直接拿旧的，
    # 磁盘上改了也不会重读。不清的话典型症状就是"明明改了却没效果"，
    # 只能重启 Maya。
    #
    # 用 pop(name, None) 而不是 del：某个模块本来就没被导入过（比如
    # planeCollider 目前根本没被 import）时不会 KeyError。
    #
    # 注意列表里**不包含 colliders_Node 自己** —— 正在执行的模块不能自己把自己
    # 从缓存里删掉，而且 _registered 这个模块级状态还要留到下次 load。
    # 本文件自身改动后，还是得靠 Maya 重新执行这个 .py 才生效。
    # 清掉模块缓存，下次加载插件时会重新读取磁盘上的代码
    for name in _SIBLING_MODULES:
        sys.modules.pop(name, None)
