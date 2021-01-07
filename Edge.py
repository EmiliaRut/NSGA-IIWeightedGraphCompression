class Edge:

    def __init__(self, n, w):
        self.node = n
        self.weight = w

    def getNode(self):
        return self.node

    def getWeight(self):
        return self.weight

    def setWeight(self, w):
        self.weight = w

    def printEdge(self):
        return "Node: %s Weight: %s" %(self.node, self.weight)