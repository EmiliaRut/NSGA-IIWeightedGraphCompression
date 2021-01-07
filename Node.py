class Node:

    def __init__(self, nodeName, nodeID):
        self.nodeName = nodeName
        self.nodeID = nodeID
        self.mergedNodes = []
        self.fakeEdges = []

    def updateID(self, upID):
        self.nodeID = upID

    def getID(self):
        return self.nodeID

    def getName(self):
        return self.nodeName

    def getMergedNodes(self):
        return self.mergedNodes

    def getFakeEdges(self):
        return self.fakeEdges

    def absorbNode(self, n):
        self.mergedNodes.append(n)

    def absorbAllNodes(self, nodes):
        for n in nodes:
            self.absorbNode(n)