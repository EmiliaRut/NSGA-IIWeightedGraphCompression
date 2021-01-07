import threading
import time


class GAThread(threading.Thread):

    def __init__(self, n, med):
        threading.Thread.__init__(self)
        self.name = n
        self.mediator = med

    def run(self):
        print(self.name + " BEGIN")
        ga = self.mediator.getGA()
        while ga is not None:
            print(self.name + " GA BEGIN")
            start = time.time()
            ga.run()
            end = time.time()
            print("GA Time: " + str(end - start))
            print(self.name + " GA COMPLETE")
            ga = self.mediator.getGA()
        print(self.name + " COMPLETE")
