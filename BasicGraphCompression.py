from Graph import Graph

import random
import numpy
import xlsxwriter
import time
import threading

from deap import base
from deap import creator
from deap import tools


class GraphCompression:
    # FILE PATHS#
    DAT_FILES = "./Files/In/"
    DATA_FILES = "./Files/Data/"
    OUTPUT_FILES = "./Files/Out/"

    def __init__(self, filename):
        # GRAPH OBJECT #
        self.currentGraph = None
        self.originalGraph = None

        # CONSTANTS FOR GENETIC ALGORITHM #
        self.OUTPREFIX = None
        self.SOURCE = None
        self.MAXGEN = None
        self.POPSIZE = None
        self.RUNS = None
        self.MUTPB = None
        self.CXPB = None
        self.MAXDIST = None
        self.FITNESS = None

        # OUPUT FILES
        self.workbook = None
        self.worksheet = None
        self.headers = None
        self.row = None

        # SAVING INFO TO PRINT TO EXCEL
        self.globalBestFit = None
        self.globalBestChrom = None
        self.runBestFit = None
        self.runBestChrom = None
        self.genBestFit = None
        self.genBestChrom = None
        self.globalWorstFit = None
        self.runWorstFit = None
        self.genWorstFit = None

        # Toolbox used for DEAP
        self.toolbox = base.Toolbox()

        self.createDEAPtoolbox()
        self.readInParameters(self.DAT_FILES + filename)
        self.loadGraph(self.DATA_FILES + self.SOURCE)
        self.printParams()

    def run(self):

        self.row = 0
        self.createExcel()

        for x in range(int(self.RUNS)):
            self.main(x)

        # close the workbook
        self.workbook.close()

    def createDEAPtoolbox(self):
        # CREATE THE CLASSES WE NEED IN DEAP
        creator.create("FitnessMin", base.Fitness, weights=(+1.0, - 1.0))
        creator.create("Chrom", numpy.ndarray, fitness=creator.FitnessMin)

        # create all the tools we need in DEAP
        self.toolbox.register("merges", self.initChrom, creator.Chrom)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.merges)
        self.toolbox.register("evaluate", self.evaluateChrom)
        # toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("mate", self.onePointCross)
        self.toolbox.register("mutate", self.mutateChrom)
        self.toolbox.register("select", tools.selNSGA2)

    def readInParameters(self, paramFile):
        # This method reads the dat file to initialize all the GA constants

        datFile = open(paramFile, "r")
        for line in datFile.readlines():
            line = line.split(" ")
            if line[0].strip() == "generations":
                self.MAXGEN = line[1].strip()
            if line[0].strip() == "crossover":
                self.CXPB = line[1].strip()
            if line[0].strip() == "mutation":
                self.MUTPB = line[1].strip()
            if line[0].strip() == "population":
                self.POPSIZE = line[1].strip()
            if line[0].strip() == "outPrefix":
                self.OUTPREFIX = line[1].strip()
            if line[0].strip() == "maxDistance":
                self.MAXDIST = line[1].strip()
            if line[0].strip() == "runs":
                self.RUNS = line[1].strip()
            if line[0].strip() == "source":
                self.SOURCE = line[1].strip()
            if line[0].strip() == "fitness":
                self.FITNESS = line[1].strip()

    def loadGraph(self, filepath):
        # This method reads the graph details from a txt file
        data = open(filepath, "r")
        size = data.readline()
        self.originalGraph = Graph(int(size))
        self.originalGraph.loadGraphFromFile(data)
        # originalGraph.printGraph()

    def printParams(self):

        print("-------- PARAMETERS -----------")

        print("OUTPREFIX = %s" % self.OUTPREFIX)
        print("SOURCE = %s" % self.SOURCE)
        print("MAXGEN = %s" % self.MAXGEN)
        print("POPSIZE = %s" % self.POPSIZE)
        print("RUNS = %s" % self.RUNS)
        print("MUTPB = %s" % self.MUTPB)
        print("CXPB = %s" % self.CXPB)
        print("MAXDIST = %s" % self.MAXDIST)
        print("FITNESS = %s" % self.FITNESS)

        print("-------- PARAMETERS -----------")

    def createExcel(self):
        self.workbook = xlsxwriter.Workbook(
            self.OUTPUT_FILES + self.OUTPREFIX + "_" + self.FITNESS + "_dst" + self.MAXDIST + "_POPSIZE" + self.POPSIZE +
            "_mut" + self.MUTPB + "_xvr" + self.CXPB + "_run" + self.RUNS + "_gen" + self.MAXGEN + ".xlsx")
        self.worksheet = self.workbook.add_worksheet()

        # write the headers in the worksheet
        self.headers = ["Graph Size", "Population Size", "Chromosome Size", "Mutation Rate", "Crossover Rate",
                        "Maximum Distance", "Run", "Max Generation", "Current Generation", "Time to Complete (s)",
                        "Global Best", "Global Worst", "Run Best", "Run Worst", "Generation Best", "Generation Worst",
                        "Global Best Chrom", "Run Best Chrom", "Generation Best Chrom"]
        for x in range(len(self.headers)):
            self.worksheet.write(self.row, x, self.headers[x])

    def main(self, x):
        info = {}

        # initialize the population
        pop = self.toolbox.population(n=int(self.POPSIZE))
        # self.printPop(pop)

        # evaluate the population
        invalid_chrom = [chrom for chrom in pop if not chrom.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_chrom)
        for ind, fit in zip(invalid_chrom, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = self.toolbox.select(pop, int(self.POPSIZE))

        # initialize the generation counter
        g = 0

        if self.runBestChrom is None:
            self.globalBestChrom = tools.selBest(pop, 1)[0]
            self.globalBestFit = self.globalBestChrom.fitness.values
            self.globalWorstFit = tools.selWorst(pop, 1)[0].fitness.values

        self.runBestChrom = tools.selBest(pop, 1)[0]
        self.runBestFit = self.runBestChrom.fitness.values
        self.runWorstFit = tools.selWorst(pop, 1)[0].fitness.values

        self.genBestChrom = tools.selBest(pop, 1)[0]
        self.genBestFit = self.genBestChrom.fitness.values
        self.genWorstFit = tools.selWorst(pop, 1)[0].fitness.values

        info['Graph Size'] = self.originalGraph.getMaxSize()
        info['Population Size'] = len(pop)
        info['Chromosome Size'] = len(pop[0])
        info['Mutation Rate'] = self.MUTPB
        info['Crossover Rate'] = self.CXPB
        info['Maximum Distance'] = self.MAXDIST
        info['Run'] = x
        info['Max Generation'] = self.MAXGEN
        info['Current Generation'] = g
        info['Time to Complete (s)'] = "TBD"
        info['Global Best'] = self.globalBestFit
        info['Global Worst'] = self.globalWorstFit
        info['Run Best'] = self.runBestFit
        info['Run Worst'] = self.runWorstFit
        info['Generation Best'] = self.genBestFit
        info['Generation Worst'] = self.genWorstFit
        info['Global Best Chrom'] = self.globalBestChrom
        info['Run Best Chrom'] = self.runBestChrom
        info['Generation Best Chrom'] = self.genBestChrom

        self.printToExcel(info)

        self.createGraph(pop, x, g)

        # evolve
        while g < int(self.MAXGEN):
            start = time.time()
            info = {}
            g = g + 1  # increment the generation number
            print("-------------------------------------------------------")
            print(str(threading.current_thread().name) + " Run " + str(x) + " Generation " + str(g))
            print("-------------------------------------------------------")

            # Vary the population
            offsprings = tools.selTournamentDCD(pop, int(self.POPSIZE))
            offsprings = [self.toolbox.clone(ind) for ind in offsprings]

            # apply crossover
            for child1, child2 in zip(offsprings[::2], offsprings[1::2]):
                # if child1.fitness.values[1] != 0 and child1.fitness.values[0] != 100 and child2.fitness.values[1] != 0 and child2.fitness.values[0] != 100:
                if random.random() < float(self.CXPB):
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # apply mutation
            for mutant in offsprings:
                if random.random() < float(self.MUTPB):
                    randomIndex = random.randint(0, len(mutant) - 1)
                    self.toolbox.mutate(mutant[randomIndex])
                    del mutant.fitness.values

            # re-evaluate the invalid fitness values
            invalid_chrom = [chrom for chrom in offsprings if not chrom.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_chrom)
            for chrom, fit in zip(invalid_chrom, fitnesses):
                chrom.fitness.values = fit
            print("Re-evaluating " + str(len(invalid_chrom)) + " chromosomes.")

            # This is just to assign the crowding distance to the individuals
            # no actual selection is done
            pop = self.toolbox.select(pop + offsprings, int(self.POPSIZE))

            end = time.time()

            # update generation/run best and worst
            self.genBestChrom = tools.selBest(pop, 1)[0]
            self.genBestFit = self.genBestChrom.fitness.values
            self.genWorstFit = tools.selWorst(pop, 1)[0].fitness.values
            if self.genBestFit < self.runBestFit:
                self.runBestChrom = self.genBestChrom
                self.runBestFit = self.runBestChrom.fitness.values
                if self.runBestFit < self.globalBestFit:
                    self.globalBestChrom = self.runBestChrom
                    self.globalBestFit = self.runBestFit

            if self.genWorstFit > self.runWorstFit:
                self.runWorstFit = self.genWorstFit
                if self.runWorstFit > self.globalWorstFit:
                    self.globalWorstFit = self.runWorstFit

            info['Graph Size'] = self.originalGraph.getMaxSize()
            info['Population Size'] = len(pop)
            info['Chromosome Size'] = len(pop[0])
            info['Mutation Rate'] = self.MUTPB
            info['Crossover Rate'] = self.CXPB
            info['Maximum Distance'] = self.MAXDIST
            info['Run'] = x
            info['Max Generation'] = self.MAXGEN
            info['Current Generation'] = g
            info['Time to Complete (s)'] = end - start
            info['Global Best'] = self.globalBestFit
            info['Global Worst'] = self.globalWorstFit
            info['Run Best'] = self.runBestFit
            info['Run Worst'] = self.runWorstFit
            info['Generation Best'] = self.genBestFit
            info['Generation Worst'] = self.genWorstFit
            info['Global Best Chrom'] = self.globalBestChrom
            info['Run Best Chrom'] = self.runBestChrom
            info['Generation Best Chrom'] = self.genBestChrom
            self.printToExcel(info)

            if int(g) % (int(self.MAXGEN) / 10) == 0:
                self.createGraph(pop, x, g)

    def initChrom(self, chromCls):
        # This method will initialize a chromosome (merges)
        randSize = random.randint(1,self.originalGraph.getMaxSize()-1)
        chrom = [None] * randSize
        for x in range(randSize):
            chrom[x] = [None] * 2
            self.mutateChrom(chrom[x])
        # check for any duplications or merges in the same cluster
        return chromCls(chrom)

    def onePointCross(self, chrom1, chrom2):
        if len(chrom1) < len(chrom2):
            shortestSize = len(chrom1)
        else:
            shortestSize = len(chrom2)

        randomIndex = random.randint(0, shortestSize-1)

        for x in range(randomIndex):
            tempChrom = [chrom1[x][0], chrom1[x][1]]
            chrom1[x] = [chrom2[x][0], chrom2[x][1]]
            chrom2[x] = [tempChrom[0], tempChrom[1]]

    def mutateChrom(self, merge):
        # This method will mutate a random merge in the chrom
        randomRoot = random.randint(0, self.originalGraph.getMaxSize()-1)
        merge[0] = randomRoot

        if self.MAXDIST == -1:
            randomNeighbor = random.randint(0, self.originalGraph.getSize()-1)
        else:
            randomRootBFSNeighbors = self.originalGraph.bfs(randomRoot, self.MAXDIST)
            randomNeighbor = randomRootBFSNeighbors[random.randint(0, len(randomRootBFSNeighbors)-1)]

        # randomOffset = math.fmod(randomNeighbor - randomRoot, originalGraph.getMaxSize())
        merge[1] = randomNeighbor

    def evaluateChrom(self, chrom):
        #This method will evaluate the solution of merges in chrom
        self.currentGraph = Graph(self.originalGraph.size)
        self.currentGraph.deepCopy(self.originalGraph)

        for m in chrom: # for every merge in chrom
            tempMerge = [m[0], m[1]]
            if self.MAXDIST == -1:
                while(self.duplicateGene(chrom, tempMerge) or self.currentGraph.sameCluster(tempMerge)):
                    # tempMerge[0] = m[0]
                    tempMerge[1] = random.randint(0, self.originalGraph.getSize()-1)
            else:
                if self.duplicateGene(chrom, tempMerge) or self.currentGraph.sameCluster(tempMerge):
                    possibleNeighbors = self.currentGraph.bfs(tempMerge[0], self.MAXDIST)
                    for n in possibleNeighbors:
                        # tempMerge[0] = m[0]
                        tempMerge[1] = n
                        if not self.duplicateGene(chrom, tempMerge) and not self.currentGraph.sameCluster(tempMerge):
                            break
                    while self.duplicateGene(chrom, tempMerge) or self.currentGraph.sameCluster(tempMerge):
                        tempMerge[0] = random.randint(0, self.originalGraph.getMaxSize()-1)
                        possibleNeighbors = self.currentGraph.bfs(tempMerge[0], self.MAXDIST)
                        # print("Length of possible neighbors: %s" % len(possibleNeighbors))
                        if len(possibleNeighbors) > 0:
                            randIndex = random.randint(0, len(possibleNeighbors)-1)
                            # print("randIndex: %s" % randIndex)
                            tempMerge[1] = possibleNeighbors[randIndex]
                        else:
                            # print("No neighbors")
                            continue
            m[0] = tempMerge[0]
            m[1] = tempMerge[1]
            self.currentGraph.mergeNodes(m[0], m[1])
        # self.currentGraph.printGraph()
        compRate, secFit = self.currentGraph.getFitness(self.FITNESS, self.originalGraph, len(chrom))
        return compRate, secFit

    def duplicateGene(self, chrom, gene):
        duplicate = False
        for g in chrom:
            if g[0] == gene[0] and g[1] == gene[1]:
                if duplicate:
                    return True
                else:
                    duplicate = True
        return False

    def printPop(self, p):
        print("------ Population ------")
        for chrom in p:
            for x in range(len(chrom)):
                print("[" + str(chrom[x][0]) + ", " + str(chrom[x][1]) + "], ", end="")
            print("")
        print("------ Population ------")

    def printChrom(self, chrom):
        for x in range(len(chrom)):
            print("[" + str(chrom[x][0]) + ", " + str(chrom[x][1]) + "], ", end="")
        print("")

    def printToExcel(self, info):
        self.row += 1

        for x in range(len(info)):
            self.worksheet.write(self.row, x, str(info[self.headers[x]]))

    def createGraph(self, pop, run, gen):
        chart = self.workbook.add_chart({'type': 'scatter'})
        chart.set_y_axis({'name': 'compression rate'})
        chart.set_x_axis({'name': self.FITNESS})
        chart.set_title({'name': 'Pareto 1 Front'})

        #write down population data in Column AA
        x_start_loc = [(len(pop)-1)*run, 26+gen]
        x_end_loc = [(len(pop)-1)*(run+1), 26+gen]
        y_start_loc = [(len(pop)-1)*run, 27+gen]
        y_end_loc = [(len(pop)-1)*(run+1), 27+gen]
        self.worksheet.write_column(*x_start_loc, data=[chrom.fitness.values[0] for chrom in pop])
        self.worksheet.write_column(*y_start_loc, data=[chrom.fitness.values[1] for chrom in pop])

        chart.add_series({
            'values': [self.worksheet.name] + x_start_loc + x_end_loc,
            'categories': [self.worksheet.name] + y_start_loc + y_end_loc,
            'name': 'Generation ' + str(gen)
        })

        self.worksheet.insert_chart(x_start_loc[0], x_start_loc[1], chart)
        self.stats(pop, x_start_loc[0], x_start_loc[1])

    def stats(self, pop, row, col):
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values for ind in pop]
        # print("fits: " + str(fits))

        length = len(pop)
        meanFit1 = sum(fit[0] for fit in fits) / length #comprate
        meanFit2 = sum(fit[1] for fit in fits) / length #secFit
        sum1 = sum((fit[0] - meanFit1) ** 2 for fit in fits)
        sum2 = sum((fit[1] - meanFit2) ** 2 for fit in fits)
        std1 = (sum1 / length) ** 0.5
        std2 = (sum2 / length) ** 0.5

        #write labels
        self.worksheet.write(row, col+2, "Length of Pop")
        self.worksheet.write(row+1, col+2, "Mean Fit 1")
        self.worksheet.write(row+2, col+2, "Mean Fit 2")
        self.worksheet.write(row+3, col+2, "Sum Fit 1")
        self.worksheet.write(row+4, col+2, "Sum Fit 2")
        self.worksheet.write(row+5, col+2, "Standard Dev 1")
        self.worksheet.write(row+6, col+2, "Standard Dev 2")

        #write values
        self.worksheet.write(row, col+3, length)
        self.worksheet.write(row+1, col+3, meanFit1)
        self.worksheet.write(row+2, col+3, meanFit2)
        self.worksheet.write(row+3, col+3, sum1)
        self.worksheet.write(row+4, col+3, sum2)
        self.worksheet.write(row+5, col+3, std1)
        self.worksheet.write(row+6, col+3, std2)
