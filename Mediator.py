class Mediator:

    def __init__(self, gaList):
        self.gas = []
        for ga in gaList:
            self.gas.append(ga)

    def getGA(self):
        if len(self.gas) != 0:
            return self.gas.pop(0)
        return None
