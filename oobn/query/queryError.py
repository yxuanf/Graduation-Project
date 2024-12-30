class NO_STATE_Error(Exception):
    """
    找不到指定的状态
    """
    def __init__(self, node: str, state: str) -> None:
        super().__init__()
        self.node = node
        self.state = state

    def __str__(self) -> str:
        return f"{self.node} has no state as {self.state}"


class NO_InputNode_Error(Exception):
    """
    没有对应的
    """
    def __init__(self, node: str, interface: str) -> None:
        super().__init__()
        self.interface = interface

    def __str__(self) -> str:
        return f"This oobn has no interface as {self.interface}"
