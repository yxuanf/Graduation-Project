from oobn.build.encapsulation import bnClass


class BuildUtils(object):
    """
    @ 建模工具类
    """

    def __init__(self):
        pass

    @staticmethod
    def getTopParent(BN: bnClass):
        """
        @ description: 返回该类的顶层父类
        Args:
            BN: 传入的类

        Returns: 该类的顶层父类

        """
        if BN.parent is None:
            return BN
        else:
            return BN.parent.getTopParent()

