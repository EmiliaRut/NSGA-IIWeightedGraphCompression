from Edge import Edge


class Neighbors:
    def __init__(self):
        self.neighborList = []

    def getNeighborList(self):
        return self.neighborList

    def addEdge(self, e):
        self.addNodeWeight(e.getNode(), e.getWeight())

    def addNodeWeight(self, n, w):
        if not self.containsNode(n):
            newEdge = Edge(n, w)
            self.neighborList.append(newEdge)

    def addAllEdges(self, edges):
        for n in edges:
            self.addEdge(n)

    def containsNode(self, n):
        found = False
        for neighbor in self.neighborList:
            if neighbor.getNode() == n:
                found = True
        return found

    def removeAllNodes(self, neighbors):
        for edge in neighbors:
            self.removeNode(edge.getNode())

    def removeNode(self, n):
        for neighbor in self.neighborList:
            if neighbor.getNode() == n:
                self.neighborList.remove(neighbor)

    def weightOf(self, n):
        for neighbor in self.neighborList:
            if neighbor.getNode() == n:
                return neighbor.getWeight()
        return 0

    def setWeightOfNode(self, n, w):
        for neighbor in self.neighborList:
            if neighbor.getNode() == n:
                neighbor.setWeight(w)

    def printNeighbors(self):
        for edge in self.neighborList:
            print("(" + edge.printEdge() + "), ", end=' ')