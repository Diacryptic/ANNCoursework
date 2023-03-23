#Importing in the Prerequisite Modules
import dataPreprocessor, random, math, json, classes
######################################














#MAIN ALGORITHM IN HERE
def GenerateANN(dataSet, minMax, limit, amountOfHiddenNodes=5, useLinearInOutput=False, useTanh=False, useMomentum=False, useBoldDriver=False, useWeightDecay=False, useAnnealing=False, useBatchLearning=False):

    #Initialising Bold Driver or Annealing
    if (useAnnealing and useBoldDriver):
        user_input = int(input("Both Bold Driver and Annealing are enabled. Please choose only one. (1 = Bold Driver / 2 = Annealing)"))
        if user_input == 1:
            useAnnealing = False
        elif user_input == 2:
            useBoldDriver = False

    if (useAnnealing):
        startRho = float(input("Please set a start value for Rho: "))
        endRho = float(input("Please set an end value for Rho: "))
    ######################################



    ##############################
    ###### USEFUL FUNCTIONS ######
    ##############################

    #Procedure to set the input nodes 'u' value to the appropriate inputs
    def InsertInputs(dataPoint):
        for i in range(len(layers[0])):
            layers[0][i].u = dataPoint[i]

    #Procedure to calculate the 'u' value of each node with the current inputs
    def CalcOutput():
        for i in range(1, len(layers)): #For each layer in the network, from the hidden layer to the output layer...

            for Node in layers[i]: #For each node in the current layer...

                temp = 0

                for prevNode in layers[i-1]: #For each node in the previous layer...
                    temp += prevNode.u * adjMatrix[prevNode.number][Node.number] #Add up each of the 'predecessor' nodes weighted outputs

                temp += adjMatrix[Node.number][Node.number] #Add the nodes bias

                Node.S = temp #Set the resulting sum as the 'S' value of the node

                #Perform the Transfer function on the new 'S' value to generate a new 'u' value for the node
                if (i == len(layers)-1 and useLinearInOutput):
                    Node.u = Node.S
                elif (useTanh): #If we are using tanh as Transfer...
                    Node.u = ((math.e**Node.S)-(math.e**-Node.S))/((math.e**Node.S)+(math.e**-Node.S))
                else: #If we are using a normal Sigmoid function
                    Node.u = 1/(1+(math.e**(-Node.S)))

    #Procedure to calculate each node's delta value
    def CalcDelta(dataPoint):
        for i in range(len(layers)-1, 0, -1): #For each layer in the network, from the output layer to the hidden layer...

            for j in range(len(layers[i])): #For each node in the current layer

                if i == len(layers)-1: #First, calculate the output node's delta value...
                    
                    if (useWeightDecay): #If we are using weight decay...

                        upsilon = 1/(rho*epochsSoFar)

                        Omega = 0
                        counter = 0
                        
                        for x in range(len(adjMatrix)):
                            for y in range(len(adjMatrix[i])):
                                if adjMatrix[x][y] != None:
                                    Omega += adjMatrix[x][y]**2
                                    counter += 1

                        Omega /= 2*counter

                        if (useLinearInOutput):
                            layers[i][j].delta = dataPoint[5] - layers[i][j].u + (upsilon*Omega)
                        elif (useTanh):
                            layers[i][j].delta = (dataPoint[5] - layers[i][j].u + (upsilon*Omega)) * (1 - layers[i][j].u**2)
                        else:
                            layers[i][j].delta = (dataPoint[5] - layers[i][j].u + (upsilon*Omega)) * (layers[i][j].u * (1-layers[i][j].u))
                    
                    else: #If not...
                        if (useLinearInOutput):
                            layers[i][j].delta = dataPoint[5] - layers[i][j].u
                        elif (useTanh):
                            layers[i][j].delta = (dataPoint[5] - layers[i][j].u) * (1 - layers[i][j].u**2)
                        else:
                            layers[i][j].delta = (dataPoint[5] - layers[i][j].u) * (layers[i][j].u * (1-layers[i][j].u))


                else: #Then, calculate all the hidden node deltas       
                    if(useTanh):
                        layers[i][j].delta = adjMatrix[layers[i][j].number][layers[i+1][0].number] * layers[i+1][0].delta * (1-(layers[i][j].u**2))
                    else:
                        layers[i][j].delta = adjMatrix[layers[i][j].number][layers[i+1][0].number] * layers[i+1][0].delta * (layers[i][j].u*(1-layers[i][j].u))

    #Procedure for updating the weights between nodes
    def UpdateWeights():
        for i in range(len(layers)-1): #From the input layer to the hidden layer (inclusive)...
            for j in range(len(layers[i])): #...grab each node individually...
                Node = layers[i][j]
                for nextNode in layers[i+1]: #...for every node in front of it, update the weight between them
                    #If using Momentum...
                    if (useMomentum):
                        oldWeight = adjMatrix[Node.number][nextNode.number]
                        newWeight = adjMatrix[Node.number][nextNode.number] + (rho * nextNode.delta * Node.u) + (0.9 * momentumMatrix[Node.number][nextNode.number])

                        adjMatrix[Node.number][nextNode.number] = newWeight

                        momentumMatrix[Node.number][nextNode.number] = newWeight - oldWeight
                    else:
                        adjMatrix[Node.number][nextNode.number] += rho * nextNode.delta * Node.u
                        
    #Procedure for updating the biases of each node
    def UpdateBiases():
        for i in range(1, len(layers)): #From the hidden layer to the output layer (inclusive)...
            for j in range(len(layers[i])): #...grab each node individually and update their bias on the weight matrix
                Node = layers[i][j]
                #If using Momentum...
                if (useMomentum): 
                    oldWeight = adjMatrix[Node.number][Node.number]
                    newWeight = adjMatrix[Node.number][Node.number] + (rho * Node.delta) + (0.9 * momentumMatrix[Node.number][Node.number])
                    
                    momentumMatrix[Node.number][Node.number] = newWeight - oldWeight
                     
                    adjMatrix[Node.number][Node.number] = newWeight
                else:
                    adjMatrix[Node.number][Node.number] += rho * Node.delta


    #Defining the Forward/Backward Pass
    def ForwardPass(dataPoint): 
        ''' 
            Forward Pass is simply inputting values, then calculating the output 
        '''
        InsertInputs(dataPoint)
        CalcOutput()

    def BackwardPass(dataPoint):
        '''
            Backward Pass is taking in the datapoint, comparing the observed output 
            to the modelled output and updating the weights and biases
        '''
        CalcDelta(dataPoint)
        UpdateWeights()
        UpdateBiases()
    ###################################

    ############################
    ##### USEFUL FUNCTIONS #####
    ############################







    ####################################
    ##### BATCH LEARNING FUNCTIONS #####
    ####################################

    def BatchEditWeights():
        for j in range(len(layers)-1):
            for Node in layers[j]:
                for nextNode in layers[j+1]:
                    if (useMomentum):
                        oldWeight = adjMatrix[Node.number][nextNode.number]
                        newWeight = adjMatrix[Node.number][nextNode.number] + (rho * batchArray[Node.number][nextNode.number]/len(dataSet)) + (0.9 * momentumMatrix[Node.number][nextNode.number])

                        adjMatrix[Node.number][nextNode.number] = newWeight

                        momentumMatrix[Node.number][nextNode.number] = newWeight - oldWeight
                    else:
                        adjMatrix[Node.number][nextNode.number] += rho * batchArray[Node.number][nextNode.number]/len(dataSet)
        
        for j in range(1, len(layers)):
            for Node in layers[j]:
                if (useMomentum):
                    oldWeight = adjMatrix[Node.number][Node.number]
                    newWeight = adjMatrix[Node.number][Node.number] + (rho * batchArray[Node.number][Node.number]/len(dataSet)) + (0.9 * momentumMatrix[Node.number][Node.number])
                    
                    momentumMatrix[Node.number][Node.number] = newWeight - oldWeight
                    
                    adjMatrix[Node.number][Node.number] = newWeight
                else:
                    adjMatrix[Node.number][Node.number] += rho * batchArray[Node.number][Node.number]/len(dataSet)

    def BatchCalcDelta(dataPoint):
        #Calculate the Delta of each node
        CalcDelta(dataPoint)

        #For each of the edges, add onto the corresponding batch array entry j's delta and i's u
        for j in range(len(layers)-1):
            for node in layers[j]:
                for nextNode in layers[j+1]:
                    batchArray[node.number][nextNode.number] += nextNode.delta * node.u
        
        #For each of the biases, add onto the corresponding batch array entry i's delta
        for j in range(1, len(layers)):
            for node in layers[j]:
                batchArray[node.number][node.number] += node.delta
    
    ####################################
    ##### BATCH LEARNING FUNCTIONS #####
    ####################################








    #######################################
    ##### INITIALISING NEURAL NETWORK #####
    #######################################

    #Setting the Learning Rate to 0.1 initially
    rho = 0.1

    #Defining how many nodes in total the network has (n inputs + m hidden nodes + 1 output)
    numberOfNodes = len(dataSet[0])+amountOfHiddenNodes

    #Create the individual nodes in the network  (inputs -> hidden layer -> output node)
    layers = [[classes.inputNode(i) for i in range(len(dataSet[0])-1)],
              
            [classes.node(i) for i in range(len(dataSet[0])-1, numberOfNodes-1)],

            [classes.node(numberOfNodes-1)]]

    #Create Weight/Bias Matrix
    adjMatrix =[[None for i in range(numberOfNodes)] for j in range(numberOfNodes)]

    #Create Momentum Matrix (stores the weight changes)
    if (useMomentum):
        momentumMatrix = [[0 for i in range(numberOfNodes)] for j in range(numberOfNodes)]
    
    #Initialise Weights
    for i in range(len(layers)-1):
        for j in range(len(layers[i])):
            thisNode = layers[i][j]
            for k in range(len(layers[i+1])):
                nextNode = layers[i+1][k]
                adjMatrix[thisNode.number][nextNode.number] = random.uniform(-2/(len(dataSet[0])-1),2/(len(dataSet[0])-1))         

    #Initialise Biases
    for i in range(1, len(layers)):
        for j in range(len(layers[i])):
            thisNode = layers[i][j]
            adjMatrix[thisNode.number][thisNode.number] = random.uniform(-2/(len(dataSet[0])-1),2/(len(dataSet[0])-1))

    #If using Bold Driver, take an initial error reading
    if (useBoldDriver):
        MSE = 0
        for dataPoint in dataSet:
            ForwardPass(dataPoint)

            observed = (dataPreprocessor.DestandardiseResult(layers[2][0].u, minMax))
            modelled = dataPreprocessor.DestandardiseResult(dataPoint[5],minMax)
            MSE += (observed-modelled)**2
        MSE = MSE / len(dataSet)

    #Keep track of how many epochs have currently been completed (for Weight Decay)
    epochsSoFar = 0

    #######################################
    ##### INITIALISING NEURAL NETWORK #####
    #######################################






    ######################################
    ##### BACKPROPAGATION ALGORITHM ######
    ######################################

    #If we are using Batch Learning
    if (useBatchLearning):

        #Loop X amount of times
        for i in range(limit):

            #Initialise 2D array to store sums of delta_j * u_i (row i and col j)
            batchArray = [[0 for x in range(numberOfNodes)] for y in range(numberOfNodes)]

            #If using Bold Driver, every so often take an error reading and compare to previous error reading
            if (useBoldDriver and i % (limit//30) == 0 and i != 0):
                oldMSE = MSE

                oldWeights = adjMatrix

                if (useMomentum):
                    oldMomentum = momentumMatrix
                
                #Perform a Forward Pass, summing delta and u
                for dataPoint in dataSet:
                    ForwardPass(dataPoint)
                    BatchCalcDelta(dataPoint)

                #Edit the weights of the network with the sums
                BatchEditWeights()

                #Calc new error reading
                MSE = 0
                for dataPoint in dataSet:
                    ForwardPass(dataPoint)
                    
                    observed = (dataPreprocessor.DestandardiseResult(layers[2][0].u, minMax))
                    modelled = dataPreprocessor.DestandardiseResult(dataPoint[5],minMax)
                    MSE += (observed-modelled)**2

                MSE = MSE/len(dataSet)

                #While the New Error is worse than the old one, reset the weights/biases, decrease the learning rate and rerun the cycle
                while (MSE > (oldMSE * 1.04)):
                    print("Learning Rate Too Large. Decreasing by 30 percent and Rerunning...")
                    rho = 0.7 * rho

                    if (rho < 0.01):
                        rho = 0.01
                        print(f"Rho: {rho}")
                        break

                    print(f"Rho: {rho}")

                    #Reset network conditions
                    adjMatrix = oldWeights

                    if (useMomentum):
                        momentumMatrix = oldMomentum

                    batchArray = [[0 for x in range(numberOfNodes)] for y in range(numberOfNodes)]
                    ########################

                    #Rerun cycle
                    for dataPoint in dataSet:
                        ForwardPass(dataPoint)
                        BatchCalcDelta(dataPoint)


                    BatchEditWeights()
                    ##############

                    #Take another error reading
                    MSE = 0
                    for dataPoint in dataSet:
                        ForwardPass(dataPoint)
                        
                        observed = (dataPreprocessor.DestandardiseResult(layers[2][0].u, minMax))
                        modelled = dataPreprocessor.DestandardiseResult(dataPoint[5],minMax)
                        MSE += (observed-modelled)**2

                    MSE = MSE/len(dataSet)
                    ###########################

                #If the New Error is signifcantly better than the old one, increase the learning rate
                if (MSE < oldMSE * 0.96):
                    print("Error Function Decreased. Increasing Learning Rate by 5%...")
                    rho = 1.05 * rho
                    if (rho > 0.5):
                        rho = 0.5
                    print(f"Rho: {rho}")

            #When we aren't using / at a Bold Driver Interval, 
            #cycle through the data set and change weights acoordingly at the end
            else:
                
                #Perform a Forward Pass, summing delta and u
                for dataPoint in dataSet:
                    ForwardPass(dataPoint)
                    BatchCalcDelta(dataPoint)

                #Edit the weights of the network with the sums
                BatchEditWeights()

            #If we are using Annealing, reduce rho accordingly
            if (useAnnealing):
                rho = endRho + ((startRho-endRho) * (1 - (1 / (1 + math.e ** (10 - (20*i/limit))))))
                print(f"Rho: {rho}")

            #Progress Identifier
            print(f"Training at: {i/limit*100}%", end="\r")

    #If we aren't Batch Learning...
    else:
        #Train the ANN
        for i in range(limit):
            epochsSoFar += 1

            #If using Bold Driver, every so often take an error reading and compare to previous error reading
            if (useBoldDriver and i % (limit//30) == 0 and i != 0):
                oldMSE = MSE

                oldWeights = adjMatrix

                if (useMomentum):
                    oldMomentum = momentumMatrix

                for dataPoint in dataSet:
                    ForwardPass(dataPoint)
                    BackwardPass(dataPoint)

                MSE = 0
                for dataPoint in dataSet:
                    ForwardPass(dataPoint)
                    
                    observed = (dataPreprocessor.DestandardiseResult(layers[2][0].u, minMax))
                    modelled = dataPreprocessor.DestandardiseResult(dataPoint[5],minMax)
                    MSE += (observed-modelled)**2

                MSE = MSE/len(dataSet)

                #While the New Error is worse than the old one, reset the weights/biases, decrease the learning rate and rerun the cycle
                while (MSE > (oldMSE * 1.04)):
                    print("Learning Rate Too Large. Decreasing by 30 percent and Rerunning...")
                    rho = 0.7 * rho

                    if (rho < 0.01):
                        rho = 0.01
                        print(f"Rho: {rho}")
                        break

                    print(f"Rho: {rho}")

                    adjMatrix = oldWeights

                    if (useMomentum):
                        momentumMatrix = oldMomentum

                    for dataPoint in dataSet:
                        ForwardPass(dataPoint)
                        BackwardPass(dataPoint)

                    MSE = 0
                    for dataPoint in dataSet:
                        ForwardPass(dataPoint)
                        
                        observed = (dataPreprocessor.DestandardiseResult(layers[2][0].u, minMax))
                        modelled = dataPreprocessor.DestandardiseResult(dataPoint[5],minMax)
                        MSE += (observed-modelled)**2

                    MSE = MSE/len(dataSet)
                
                #If the New Error is signifcantly better than the old one, increase the learning rate
                if (MSE < oldMSE * 0.96):
                    print("Error Function Decreased. Increasing Learning Rate by 5%...")
                    rho = 1.05 * rho
                    if (rho > 0.5):
                        rho = 0.5
                    print(f"Rho: {rho}")

            #If not, train as usual
            else:
                for dataPoint in dataSet:
                    ForwardPass(dataPoint)
                    BackwardPass(dataPoint)

            if (useAnnealing):
                rho = endRho + ((startRho-endRho) * (1 - (1 / (1 + math.e ** (10 - (20*i/limit))))))
                print(f"Rho: {rho}")
            
            #Progress Identifier
            print(f"Training at: {i/limit*100}%", end="\r")

    

    ######################################
    ##### BACKPROPAGATION ALGORITHM ######
    ######################################




    print(end="\n\n")

    #Encapsulate the ANN into an object then return
    return classes.ANN(layers,adjMatrix, useLinearInOutput, useTanh, limit, useMomentum, useBoldDriver, useWeightDecay, useAnnealing, useBatchLearning)
###################################

def saveANN(ann, RMSE):
    file = open(f"{ann}.json", "w")
    annJSON = json.dumps(ann.save(RMSE), indent=4)
    file.write(annJSON)
    file.close()


def saveBestANN(ann,RMSE):
    file = open(f"ANN.json", 'r')
    data = json.load(file)
    if RMSE < data['RMSE']:
        print("Best ANN So Far!")
        file.close()
        file = open(f"ANN.json", 'w')
        annJSON = json.dumps(ann.save(RMSE), indent=4)
        file.write(annJSON)
    file.close()
    











#Main Executable Function
def main(epochs, numberOfHiddenNodes, useLinearInOutput, useTanh, useMomentum, useBoldDriver, useWeightDecay, useAnnealing, useBatchLearning, GenerateNewFile=False, amountOfANNs=1, randomConfig=False):

    #Read or Generate the Data Sets
    if (GenerateNewFile):
        training, validation, testing, minMax = dataPreprocessor.GenerateDatasets("Formatted DS.csv")
    else:
        try:
            training, validation, testing, minMax = dataPreprocessor.ReadDatasetJSON("dataset.json")
        except:
            print("No Saved Dataset Found. Generating New Dataset...")
            training, validation, testing, minMax = dataPreprocessor.GenerateDatasets("Formatted DS.csv")

    #Generate x number of ANNs with the specified config (i.e. x epochs, y hidden nodes, useMomentum etc.)
    annArray = []
    for i in range(amountOfANNs):
        if(randomConfig):
            annArray.append(GenerateANN(training, minMax, random.randint(100,10_000), random.randint(1,10), bool(random.getrandbits(1)), bool(random.getrandbits(1)), bool(random.getrandbits(1)), bool(random.getrandbits(1)), bool(random.getrandbits(1))))
        else:
            annArray.append(GenerateANN(training, minMax, epochs, numberOfHiddenNodes, useLinearInOutput, useTanh, useMomentum, useBoldDriver, useWeightDecay, useAnnealing, useBatchLearning))
        print(f"Progress: {i+1}/{amountOfANNs} ANN generated.")
    
    #Evaluate which generated ANN performs the best against an unseen Validation Set
    bestRMSE = 9999999999999
    for ann in annArray:
        RMSE = 0
        for datapoint in validation:
            predicted = dataPreprocessor.DestandardiseResult(ann.ForwardPass(datapoint), minMax)
            observed = dataPreprocessor.DestandardiseResult(datapoint[5], minMax)
            RMSE += (observed - predicted)**2
        RMSE = math.sqrt(RMSE/len(validation))
        if RMSE < bestRMSE:
            bestRMSE = RMSE
            bestANN = ann
        print(ann, "| RMSE:", RMSE)
    
    #Save the best performing ANN out of the generate set
    print(f"Best ANN: |{bestANN}|")
    print("Saving to JSON File...")
    saveANN(bestANN, bestRMSE)

    #If this is the Best ANN that we have ever produced, save it.
    saveBestANN(bestANN, bestRMSE)


main(32768,9,useLinearInOutput=False,useTanh=False,useMomentum=True,useBoldDriver=True,useWeightDecay=False, useAnnealing=True, useBatchLearning=False, amountOfANNs=1)