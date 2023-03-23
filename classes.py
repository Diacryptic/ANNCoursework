import math

#Defining the ANN node classes
class inputNode:
    def __init__(self, number):
        self.number = number
        self.u = -1

class node:
    def __init__(self, number):
        self.number = number
        self.delta = -1
        self.S = -1
        self.u = -1

#Defining the Neural Network Class
class ANN:
    def __init__(self, network, adjWeightMatrix, useLinearInOutput, useTanh, epochs, momentum, boldDriver, weightDecay, useAnnealing, useBatchLearning):
        self.network = network
        self.adjWeightMatrix = adjWeightMatrix
        self.useLinearInOutput = useLinearInOutput
        self.useTanh = useTanh
        self.epochs = epochs
        self.momentum = momentum
        self.boldDriver = boldDriver
        self.weightDecay = weightDecay
        self.annealing = useAnnealing
        self.batchLearning = useBatchLearning
    
    def __str__(self):
        return f"Using Linear Function in Output Node: {self.useLinearInOutput}, Using Tanh: {self.useTanh}, Epochs Trained: {self.epochs}, Amount of Hidden Nodes: {len(self.network[1])}, Uses Momentum: {self.momentum}, Uses Bold Driver: {self.boldDriver}, Uses Weight Decay: {self.weightDecay}, Uses Annealing: {self.annealing}, Uses Batch Learning: {self.batchLearning}"

    def ForwardPass(self, dataPoint):
        self.InsertInputs(dataPoint)
        return self.CalcOutput()

    def InsertInputs(self, dataPoint):
        for i in range(len(self.network[0])):
            self.network[0][i].u = dataPoint[i]

    def CalcOutput(self):
        for i in range(1, len(self.network)):
            for Node in self.network[i]:
                temp = 0
                for prevNode in self.network[i-1]:
                    temp += prevNode.u * self.adjWeightMatrix[prevNode.number][Node.number]
                temp += self.adjWeightMatrix[Node.number][Node.number]
                Node.S = temp
                
                if (i == len(self.network)-1 and self.useLinearInOutput):
                    Node.u = Node.S
                elif (self.useTanh):
                    Node.u = ((math.e**Node.S)-(math.e**-Node.S))/((math.e**Node.S)+(math.e**-Node.S))
                else:
                    Node.u = 1/(1+(math.e**(-Node.S)))

        return self.network[len(self.network)-1][0].u

    def save(self, RMSE):
        network = {
            "config" : (len(self.network[0]),len(self.network[1]),len(self.network[2])),
            "adjMatrix" : self.adjWeightMatrix,
            "useLinearInOutput" : self.useLinearInOutput,
            "useTanh" : self.useTanh,
            "epochs" : self.epochs,
            "useMomentum" : self.momentum,
            "useBoldDriver" : self.boldDriver,
            "useWeightDecay" : self.weightDecay,
            "useAnnealing":self.annealing,
            "useBatchLearning":self.batchLearning,
            "RMSE": RMSE
        }
        return network
    
    
##############################