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
            # originalWeight = 0
            for n in range(originalGraph.getMaxSize()):
                for e in range(originalGraph.getMaxSize()):
                    secWeight += abs(originalGraph.getGraph()[n].weightOf(e) - self.linkedGraph[n].weightOf(e))
                    # originalWeight += originalGraph.getGraph()[n].weightOf(e)

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

        masterID = self.NODES[master].getID()
        self.NODES[slave].updateID(masterID)
        for node in self.NODES[slave].getMergedNodes():
            self.NODES[node].updateID(masterID)

        #connect slave neighbors (minus master) to master
        slaveNeighbors = Neighbors()
        slaveNeighbors.addAllEdges(self.linkedGraph[slave].getNeighborList())
        for n in self.NODES[master].getMergedNodes():
            slaveNeighbors.removeNode(n)
        for n in self.NODES[slave].getMergedNodes():
            slaveNeighbors.removeNode(n)
        slaveNeighbors.removeNode(master)
        slaveNeighbors.removeNode(slave)
        for edge in slaveNeighbors.getNeighborList():
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
                        self.linkedGraph

        # connect master neighbors (minus slave) to slave
        masterNeighbors = Neighbors()
        masterNeighbors.addAllEdges(self.linkedGraph[master].getNeighborList())
        for n in self.NODES[master].getMergedNodes():
            masterNeighbors.removeNode(n)
        for n in self.NODES[slave].getMergedNodes():
            masterNeighbors.removeNode(n)
        masterNeighbors.removeNode(master)
        masterNeighbors.removeNode(slave)
        for edge in masterNeighbors.getNeighborList():
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

        # add or update an edge with new weight between slave and master
        numOfMasterMergedNodes = 1 + len(self.NODES[master].getMergedNodes())
        numOfMasterEdges = (numOfMasterMergedNodes*(numOfMasterMergedNodes-1)/2)
        numOfSlaveMergedNodes = 1 + len(self.NODES[slave].getMergedNodes())
        numOfSlaveEdges = (numOfSlaveMergedNodes*(numOfSlaveMergedNodes-1)/2)
        totalNumOfNodes = numOfMasterMergedNodes + numOfSlaveMergedNodes
        numOfTotalEdges = (totalNumOfNodes*(totalNumOfNodes-1)/2)
        remainingEdges = numOfTotalEdges - numOfSlaveEdges - numOfMasterEdges

        masterEdgeWeight = 0
        if len(self.NODES[master].getMergedNodes()) > 0:
            masterEdgeWeight = self.linkedGraph[master].weightOf(self.NODES[master].getAnyMergedNode())

        slaveEdgeWeight = 0
        if len(self.NODES[slave].getMergedNodes()) > 0:
            slaveEdgeWeight = self.linkedGraph[slave].weightOf(self.NODES[slave].getAnyMergedNode())

        newEdgeWeight = ((masterEdgeWeight * numOfMasterEdges) + (slaveEdgeWeight * numOfSlaveEdges) + (self.linkedGraph[master].weightOf(slave) * remainingEdges)) / numOfTotalEdges

        completeGraph = []
        completeGraph.extend(self.NODES[master].getMergedNodes())  # add master's merged nodes
        completeGraph.append(master)  # add master
        completeGraph.extend(self.NODES[slave].getMergedNodes())  # add slave's merged nodes
        completeGraph.append(slave)  # add slave

        while len(completeGraph) > 0:
            nOne = completeGraph.pop(0)
            for nTwo in completeGraph:
                if self.linkedGraph[nOne].containsNode(nTwo):
                    self.linkedGraph[nOne].setWeightOfNode(nTwo, newEdgeWeight)
                    self.linkedGraph[nTwo].setWeightOfNode(nOne, newEdgeWeight)
                else:
                    self.linkedGraph[nOne].addNodeWeight(nTwo, newEdgeWeight)
                    self.linkedGraph[nTwo].addNodeWeight(nOne, newEdgeWeight)

        # add the slave as a merged node of master
        self.NODES[master].absorbNode(slave)
        for node in self.NODES[slave].getMergedNodes():
            self.NODES[master].absorbNode(node)

        # update size of graph
        self.size -= 1

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