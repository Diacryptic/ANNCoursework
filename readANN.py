#Importing in the Prerequisite Modules
import dataPreprocessor, math, json, classes
import matplotlib.pyplot as plt
import numpy
######################################




#Takes in ANN config json file and recreates network
def recreateANN(file):
    temp = open(file,'r')
    savedNetwork = json.load(temp)
    temp.close()

    config = [[classes.inputNode(i) for i in range(savedNetwork["config"][0])],
              [classes.node(i) for i in range(savedNetwork["config"][0], savedNetwork["config"][0] + savedNetwork["config"][1])],
              [classes.node(i) for i in range(savedNetwork["config"][0] + savedNetwork["config"][1], savedNetwork["config"][0] + savedNetwork["config"][1] + savedNetwork["config"][2])]]
    
    adjMatrix = savedNetwork["adjMatrix"]

    useLinearInOutput = savedNetwork.get("useLinearInOutput")
    useTanh = savedNetwork["useTanh"]
    epochs = savedNetwork["epochs"]
    useMomentum = savedNetwork["useMomentum"]
    useBoldDriver = savedNetwork["useBoldDriver"]
    useWeightDecay = savedNetwork["useWeightDecay"]
    useAnnealing = savedNetwork.get('useAnnealing')
    useBatchLearning = savedNetwork.get('useBatchLearning')

    return classes.ANN(config, adjMatrix, useLinearInOutput, useTanh, epochs, useMomentum, useBoldDriver, useWeightDecay, useAnnealing, useBatchLearning)







training, validation, testing, minMax = dataPreprocessor.ReadDatasetJSON("dataset.json")

baseline = recreateANN("BestANN.json")




LINESTpredicted = [0.18900548,0.29174449,0.19965257,0.15215879,0.42182896,0.23046853,0.26073005,0.27328124,0.36075107,0.34723625,0.43987102,0.46402157,0.18128038,0.12390177,0.39275563,
                   0.48085041,0.41251227,0.33572934,0.29063242,0.67778974,0.4449197,0.34303077,0.15615796,0.24509336,0.29010554,0.48215994,0.24706378,0.26682917,0.25474474,0.27548074,
                   0.285105827,0.342161943,0.28103816,0.158040245,0.337266816,0.543737052,0.76264479,0.679065079,0.483530528,0.216197587,0.187672394,0.248835938,0.279435345,0.22800732,
                   0.333680314, 0.053935485,0.515575348,0.40136791,0.389599721,0.588151994]
LINESTobserved = [0.21,0.27,0.21,0.16,0.4,0.24,0.28,0.28,0.36,0.34,0.4,0.46,0.21,0.17,0.42,
                  0.5,0.4,0.35,0.3,0.72,0.44,0.33,0.18,0.23,0.28,0.42,0.28,0.28,0.26,0.28,
                  0.3,0.32,0.27,0.18,0.32,0.51,0.85,0.81,0.49,0.25,0.19,0.24,0.26,0.21,0.3,
                  0.14,0.5,0.36,0.4,0.53]


predictedArray = []
observedArray = []

observedMean = 0
for datapoint in testing:
    observedMean += dataPreprocessor.DestandardiseResult(datapoint[5],minMax)
observedMean /= len(testing)

CENum = 0
CEDenom = 0

MSRE = 0
RMSE = 0
for i in range(len(testing)):

    datapoint = testing[i]

    predicted = dataPreprocessor.DestandardiseResult(baseline.ForwardPass(datapoint), minMax)
    observed = dataPreprocessor.DestandardiseResult(datapoint[5], minMax)

    predictedArray.append(predicted)
    observedArray.append(observed)

    RMSE += (predicted-observed)**2

    MSRE += ((predicted-observed)/observed)**2

    CENum += (predicted-observed)**2
    CEDenom += (observed-observedMean)**2

CE = 1 - (CENum/CEDenom)
MSRE /= len(testing)
RMSE = math.sqrt(RMSE/ len(testing))

print(f"Best ANN: {RMSE}")
print(f"CE of Best ANN: {CE}")
print(f"MSRE of Best ANN: {MSRE}")
print(f"Correlation Coefficient of ANN: {numpy.corrcoef(predictedArray,  observedArray)}")







CENum = 0
CEDenom = 0

MSRE = 0
RMSE = 0
for i in range(len(LINESTobserved)):
    predicted = LINESTpredicted[i]
    observed = LINESTobserved[i]

    RMSE += (predicted - observed)**2

    MSRE += ((predicted-observed)/observed)**2

    CENum += (predicted-observed)**2
    CEDenom += (observed-observedMean)**2

CE = 1 - (CENum/CEDenom)

MSRE /= len(LINESTobserved)

RMSE = math.sqrt(RMSE/len(LINESTobserved))
print(f"LINEST Excel Function: {RMSE}")
print(f"LINEST CE: {CE}")
print(f"LINEST MSRE: {MSRE}")
print(f"Correlation Coefficient of LINEST: {numpy.corrcoef(LINESTpredicted,  LINESTobserved)}")


plt.plot([0,1],[0,1])
plt.scatter(predictedArray,  observedArray, s = 0.1, c="r")
plt.scatter(LINESTpredicted,  LINESTobserved, s=1, c="g")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
