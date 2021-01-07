from BasicGraphCompression import GraphCompression
from Mediator import Mediator
from GAThread import GAThread

if __name__ == "__main__":
    # VARIABLES
    MAX_THREADS = 1
    FIRST_TEST = 1
    LAST_TEST = 1
    TESTS = ["June2_?.dat"]

    threads = [None] * MAX_THREADS
    threadedGAs = []

    for i in range(FIRST_TEST, LAST_TEST+1):
        for test in TESTS:
            filename = test.replace("?", str(i))
            print("Filename: " + filename)
            threadedGAs.append(GraphCompression(filename))

    shared = Mediator(threadedGAs)
    for t in range(MAX_THREADS):
        name = "Thread-" + str(t)
        GAThread(name, shared).start()
