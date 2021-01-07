class WrappedNode:

    def __init__(self, nodeName, dist):
        self.nodeName = nodeName
        self.dist = dist

    def compareTo(self, wrappedNode):
        if wrappedNode.dist == self.dist:
            return 0
        elif wrappedNode.dist < self.dist:
            return 1
        else:
            return -1