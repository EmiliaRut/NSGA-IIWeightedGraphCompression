from BasicGraphCompression import GraphCompression
from Mediator import Mediator
from GAThread import GAThread
from queue import Queue
import threading
import time

def runGA(g):
    while not g.empty():
        ga = g.get()
        ga.run()
        g.task_done()

if __name__ == "__main__":
    # VARIABLES
    MAX_THREADS = 1
    FIRST_TEST = 1
    LAST_TEST = 1
    RUNS = 1
    TESTS = ["July15_?.dat"]

    threads = [None] * MAX_THREADS
    GAs = Queue()

    for i in range(FIRST_TEST, LAST_TEST+1):
        for test in TESTS:
            filename = test.replace("?", str(i))
            print("Filename: " + filename)
            for x in range(RUNS):
                # name = "Thread-Test-" + str(i) + "-Run-" + str(x)
                GAs.put(GraphCompression(filename))

    for x in range(MAX_THREADS):
        worker = threading.Thread(target=runGA, args=(GAs,))
        worker.start()

    GAs.join()
    print("All runs complete")