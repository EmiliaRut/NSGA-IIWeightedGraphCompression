from Neighbors import Neighbors
from Node import Node
from WrappedNode import WrappedNode
from itertools import repeat

import math


class Graph:

    def __init__(self, s):
        self.linkedGraph = []
        self.size = s
        self.MAX_SIZE = s
        self.NODES = [None] * self.size
        for x in range(s):
            self.NODES[x] = Node(x, x)
            self.linkedGraph.append(Neighbors())

    def deepCopy(self, other):
        # for all Nodes
        for i in range(int(self.MAX_SIZE)):
            # copy the neighbors into the linkedGraph
            self.linkedGraph[i].addAllEdges(other.getGraph()[i].getNeighborList())
            # update the reference of each node
            index = other.NODES[i].getID()
            self.NODES[i].updateID(index)

    def getMaxSize(self):
        return self.MAX_SIZE

    def getCurrentSize(self):
        return self.size

    def sameCluster(self, merge):
        node1id = self.NODES[merge[0]].getID()
        node2id = self.NODES[merge[1]].getID()
        return node1id == node2id


    def addEdge(self, node1, node2, weight):
        aFrom = self.NODES[node1].getID()
        aTo = self.NODES[node2].getID()
        if aTo not in self.linkedGraph[aFrom].getNeighborList():
            self.linkedGraph[aFrom].addNodeWeight(aTo, weight)
        if aFrom not in self.linkedGraph[aTo].getNeighborList():
            self.linkedGraph[aTo].addNodeWeight(aFrom, weight)

    def loadGraphFromFile(self, data):
        for line in data.readlines():
            line = line.split("\t")
            # print("Line: " + str(line))
            masterNode = int(line[0])
            slaveNode = int(line[1])
            if len(line) == 3:
                weight = int(line[2])
            else:
                weight = 0
            self.addEdge(masterNode, slaveNode, weight)

    def printGraph(self):
        for i in range(self.MAX_SIZE):
            print("Node " + str(i) + ": ", end='')
            self.linkedGraph[i].printNeighbors()
            print("")

    def getGraph(self):
        return self.linkedGraph

    def getNeighbor(self, x):
        id = self.NODES[x].getID()
        return self.linkedGraph[id].getNeighborList()

    def getFitness(self, fit, originalGraph, lengthOfChrom):
        ## Third fitness / check
        # originalWeight = 0
        # for n in originalGraph.getGraph():
        #     for e in n.getNeighborList():
        #         originalWeight += e.getWeight()
        # compressedWeight = 0
        # for n in self.linkedGraph:
        #     for e in n.getNeighborList():
        #         compressedWeight += e.getWeight()
        # check = abs(compressedWeight - originalWeight)
        compRate = float(lengthOfChrom) / float(originalGraph.getMaxSize())

        ## First fitness // AbsWeightDiff
        if fit == "SumAbsWeightDiff":
            secWeight = 0
            originalWeight = 0
            for n in range(originalGraph.getMaxSize()):
                for e in range(originalGraph.getMaxSize()):
                    secWeight += abs(originalGraph.getGraph()[n].weightOf(e) - self.linkedGraph[n].weightOf(e))
                    originalWeight += originalGraph.getGraph()[n].weightOf(e)

        ## Second fitness // SqrWeightDiff
        elif fit == "SqrWeightDiff":
            secWeight = 0
            for n in range(originalGraph.getMaxSize()):
                for e in range(originalGraph.getMaxSize()):
                    secWeight += math.pow(originalGraph.getGraph()[n].weightOf(e) - self.linkedGraph[n].weightOf(e), 2)

            secWeight = math.sqrt(secWeight)
        elif fit == "FakeEdges":
            currentEdges = 0
            originalEdges = 0
            for n in range(originalGraph.getMaxSize()):
                currentEdges += len(self.linkedGraph[n].getNeighborList())
                originalEdges += len(originalGraph.getGraph()[n].getNeighborList())
            secWeight = currentEdges - originalEdges
        else:
            secWeight = -1

        return compRate, secWeight

    def mergeNodes(self, m, s):
        master = self.NODES[m].getID()
        slave = self.NODES[s].getID()

        if master == slave:
            print("master == slave in merge")
            return

        self.NODES[slave].updateID(self.NODES[master].getID())

        #connect slave neighbors (minus master) to master
        slaveNeighbors = Neighbors()
        slaveNeighbors.addAllEdges(self.linkedGraph[slave].getNeighborList())
        map(slaveNeighbors.removeNode, self.NODES[slave].getMergedNodes())
        map(slaveNeighbors.removeNode, self.NODES[master].getMergedNodes())
        slaveNeighbors.removeNode(master)
        slaveNeighbors.removeNode(slave)
        map(self.connectSlaveNeighbors, slaveNeighbors.getNeighborList(), repeat(master), repeat(slave))

        # connect master neighbors (minus slave) to slave
        masterNeighbors = Neighbors()
        masterNeighbors.addAllEdges(self.linkedGraph[master].getNeighborList())
        map(masterNeighbors.removeNode, self.NODES[master].getMergedNodes())
        map(masterNeighbors.removeNode, self.NODES[slave].getMergedNodes())
        masterNeighbors.removeNode(master)
        masterNeighbors.removeNode(slave)
        map(self.connectMasterNeighbors, masterNeighbors.getNeighborList(), repeat(master), repeat(slave))

        # add or update an edge with new weight between slave and master
        numOfEdges = 1
        totalWeight = self.linkedGraph[slave].weightOf(master)
        for mNode in self.NODES[master].getMergedNodes():
            totalWeight = totalWeight + self.linkedGraph[slave].weightOf(mNode)
            totalWeight = totalWeight + self.linkedGraph[master].weightOf(mNode)
            numOfEdges += 2
            for sNode in self.NODES[slave].getMergedNodes():
                totalWeight = totalWeight + self.linkedGraph[mNode].weightOf(sNode)
                numOfEdges += 1
        for sNode in self.NODES[slave].getMergedNodes():
            totalWeight = totalWeight + self.linkedGraph[master].weightOf(sNode)
            totalWeight = totalWeight + self.linkedGraph[slave].weightOf(sNode)
            numOfEdges += 2
        newEdgeWeight = totalWeight/numOfEdges

        # update or add weight between slave and master
        if self.linkedGraph[master].containsNode(slave):
            self.linkedGraph[master].setWeightOfNode(slave, newEdgeWeight)
            self.linkedGraph[slave].setWeightOfNode(master, newEdgeWeight)
        else:
            self.linkedGraph[master].addNodeWeight(slave, newEdgeWeight)
            self.linkedGraph[slave].addNodeWeight(master, newEdgeWeight)

        # for all slave merge nodes
        for sNode in self.NODES[slave].getMergedNodes():
            # update or add weight between slave merge and master
            if self.linkedGraph[master].containsNode(sNode):
                self.linkedGraph[master].setWeightOfNode(sNode, newEdgeWeight)
                self.linkedGraph[sNode].setWeightOfNode(master, newEdgeWeight)
            else:
                self.linkedGraph[master].addNodeWeight(sNode, newEdgeWeight)
                self.linkedGraph[sNode].addNodeWeight(master, newEdgeWeight)

            # update or add weight between slave merge and slave
            if self.linkedGraph[slave].containsNode(sNode):
                self.linkedGraph[slave].setWeightOfNode(sNode, newEdgeWeight)
                self.linkedGraph[sNode].setWeightOfNode(slave, newEdgeWeight)
            else:
                self.linkedGraph[slave].addNodeWeight(sNode, newEdgeWeight)
                self.linkedGraph[sNode].addNodeWeight(slave, newEdgeWeight)

            # for all master merges
            for mNode in self.NODES[master].getMergedNodes():
                # update or add weight
                if self.linkedGraph[mNode].containsNode(sNode):
                    self.linkedGraph[mNode].setWeightOfNode(sNode, newEdgeWeight)
                    self.linkedGraph[sNode].setWeightOfNode(mNode, newEdgeWeight)
                else:
                    self.linkedGraph[mNode].addNodeWeight(sNode, newEdgeWeight)
                    self.linkedGraph[sNode].addNodeWeight(mNode, newEdgeWeight)

        # for all master merges
        for mNode in self.NODES[master].getMergedNodes():
            # update or add weight between master merge and master
            if self.linkedGraph[master].containsNode(mNode):
                self.linkedGraph[master].setWeightOfNode(mNode, newEdgeWeight)
                self.linkedGraph[mNode].setWeightOfNode(master, newEdgeWeight)
            else:
                self.linkedGraph[master].addNodeWeight(mNode, newEdgeWeight)
                self.linkedGraph[mNode].addNodeWeight(master, newEdgeWeight)

            # update or add weight between master merge and slave
            if self.linkedGraph[slave].containsNode(mNode):
                self.linkedGraph[slave].setWeightOfNode(mNode, newEdgeWeight)
                self.linkedGraph[mNode].setWeightOfNode(slave, newEdgeWeight)
            else:
                self.linkedGraph[slave].addNodeWeight(mNode, newEdgeWeight)
                self.linkedGraph[mNode].addNodeWeight(slave, newEdgeWeight)

        # add the slave as a merged node of master
        self.NODES[master].absorbNode(slave)
        map(self.NODES[master].absorbNode, self.NODES[slave].getMergedNodes())

        # update size of graph
        self.size -= 1

    def connectSlaveNeighbors(self, edge, master, slave):
        # average out the edge weights
        numOfEdges = 2 + len(self.NODES[slave].getMergedNodes()) + len(self.NODES[master].getMergedNodes())
        totalWeight = (edge.getWeight() * (1 + len(self.NODES[slave].getMergedNodes()))) + (
        self.linkedGraph[master].weightOf(edge.getNode()) * (1 + len(self.NODES[master].getMergedNodes())))
        newEdgeWeight = totalWeight / numOfEdges

        # update weight between edge and slave
        self.linkedGraph[slave].setWeightOfNode(edge.getNode(), newEdgeWeight)
        self.linkedGraph[edge.getNode()].setWeightOfNode(slave, newEdgeWeight)

        # update/add weight between edge and master
        if self.linkedGraph[master].containsNode(edge.getNode()):
            self.linkedGraph[master].setWeightOfNode(edge.getNode(), newEdgeWeight)
            self.linkedGraph[edge.getNode()].setWeightOfNode(master, newEdgeWeight)
        else:
            self.linkedGraph[master].addNodeWeight(edge.getNode(), newEdgeWeight)
            self.linkedGraph[edge.getNode()].addNodeWeight(master, newEdgeWeight)

        # update or add weight between all the merged nodes in slave to edge
        for node in self.NODES[slave].getMergedNodes():
            if node != edge.getNode():
                if self.linkedGraph[edge.getNode()].containsNode(node):
                    self.linkedGraph[node].setWeightOfNode(edge.getNode(), newEdgeWeight)
                    self.linkedGraph[edge.getNode()].setWeightOfNode(node, newEdgeWeight)
                else:
                    self.linkedGraph[node].addNodeWeight(edge.getNode(), newEdgeWeight)
                    self.linkedGraph[edge.getNode()].addNodeWeight(node, newEdgeWeight)

        # update or add weight between all the merged nodes in master to edge
        for node in self.NODES[master].getMergedNodes():
            if node != edge.getNode():
                if self.linkedGraph[edge.getNode()].containsNode(node):
                    self.linkedGraph[node].setWeightOfNode(edge.getNode(), newEdgeWeight)
                    self.linkedGraph[edge.getNode()].setWeightOfNode(node, newEdgeWeight)
                else:
                    self.linkedGraph[node].addNodeWeight(edge.getNode(), newEdgeWeight)
                    self.linkedGraph[edge.getNode()].addNodeWeight(node, newEdgeWeight)

    def connectMasterNeighbors(self, edge, master, slave):
        # average out the edge weights
        numOfEdges = 2 + len(self.NODES[slave].getMergedNodes()) + len(self.NODES[master].getMergedNodes())
        totalWeight = (edge.getWeight() * (1 + len(self.NODES[master].getMergedNodes()))) + (
        self.linkedGraph[slave].weightOf(edge.getNode()) * (1 + len(self.NODES[slave].getMergedNodes())))
        newEdgeWeight = totalWeight / numOfEdges

        # update weight between edge and master
        self.linkedGraph[master].setWeightOfNode(edge.getNode(), newEdgeWeight)
        self.linkedGraph[edge.getNode()].setWeightOfNode(master, newEdgeWeight)

        # update/add weight between edge and slave
        if self.linkedGraph[slave].containsNode(edge.getNode()):
            self.linkedGraph[slave].setWeightOfNode(edge.getNode(), newEdgeWeight)
            self.linkedGraph[edge.getNode()].setWeightOfNode(slave, newEdgeWeight)
        else:
            self.linkedGraph[slave].addNodeWeight(edge.getNode(), newEdgeWeight)
            self.linkedGraph[edge.getNode()].addNodeWeight(slave, newEdgeWeight)

        # update or add weight between all the merged nodes in master to edge
        for node in self.NODES[master].getMergedNodes():
            if node != edge.getNode():
                if self.linkedGraph[edge.getNode()].containsNode(node):
                    self.linkedGraph[node].setWeightOfNode(edge.getNode(), newEdgeWeight)
                    self.linkedGraph[edge.getNode()].setWeightOfNode(node, newEdgeWeight)
                else:
                    self.linkedGraph[node].addNodeWeight(edge.getNode(), newEdgeWeight)
                    self.linkedGraph[edge.getNode()].addNodeWeight(node, newEdgeWeight)

        # update or add weight between all the merged nodes in slave to edge
        for node in self.NODES[slave].getMergedNodes():
            if node != edge.getNode():
                if self.linkedGraph[edge.getNode()].containsNode(node):
                    self.linkedGraph[node].setWeightOfNode(edge.getNode(), newEdgeWeight)
                    self.linkedGraph[edge.getNode()].setWeightOfNode(node, newEdgeWeight)
                else:
                    self.linkedGraph[node].addNodeWeight(edge.getNode(), newEdgeWeight)
                    self.linkedGraph[edge.getNode()].addNodeWeight(node, newEdgeWeight)

    def bfs(self, masterNode, dist):
        # This method will perform breadth first search to get all the neighbors dist edges away
        masterID = self.NODES[masterNode].getID()
        exploredSet = set()
        toExploreList = []
        exploredSet.add(masterID)
        toExploreList.append(WrappedNode(masterID, 0))
        #explore while there are items to explore
        while len(toExploreList) > 0:
            at = toExploreList[0]
            toExploreList.remove(at)
            atID = self.NODES[at.nodeName].getID()
            atDistance = at.dist
            if int(atDistance) < int(dist):
                atNeighbors = self.linkedGraph[atID].getNeighborList()
                for n in atNeighbors:
                    nID = self.NODES[n.getNode()].getID()
                    if nID not in exploredSet:
                        toExploreList.append(WrappedNode(n.getNode(), atDistance+1))
                        exploredSet.add(nID)
        exploredSet.remove(masterID)
        return list(exploredSet)