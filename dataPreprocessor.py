import random,math,json
import matplotlib.pyplot as plt

def ReadData(filename): 
    #Reads the data from the .csv file, shuffles the data then converts into 3 2D arrays (Training, Validation, Testing sets)
    with open(filename, 'r', encoding='utf-8-sig') as file:
        content = file.readlines()

    for i in range(len(content)):
        content[i] = (content[i].strip()).split(',')
        for j in range(len(content[i])):
            content[i][j] = float(content[i][j])

    random.shuffle(content)

    training = content[:(len(content)//10)*6]
    validation = content[(len(content)//10)*6:(len(content)//10)*8]
    test = content[(len(content)//10)*8:]

    return training, validation, test

def CalcMean(data): 
    #Calculates the mean for each attribute (to be used to later detect outliers)
    meanSet = [0 for i in range(len(data[0]))]
    for record in data:
        for i in range(len(record)):
            meanSet[i] += record[i]
    for i in range(len(meanSet)):
        meanSet[i] /= len(data)
    return meanSet

def CalcSD(data, meanSet): 
    #Calculates the Standard Deviation for each attribute (to be used to later detect outliers)
    SDSet = [0 for i in range(len(data[0]))]
    for record in data:
        for i in range(len(record)):
            SDSet[i] += (record[i]-meanSet[i])**2
    for i in range(len(SDSet)):
        SDSet[i] = math.sqrt(SDSet[i]/len(data))
    return SDSet

def removeOutliers(data, meanSet, SDSet): 
    #Function to remove any data points from the data set if they contain an attribute value that is +-3 SDs aways from the mean
    recordsToRemove = []
    for i in range(len(data)):
        record = data[i]
        for j in range(len(record)):
            if (record[j] < meanSet[j] - (3*SDSet[j])) or (record[j] > meanSet[j] + (3*SDSet[j])):
                recordsToRemove.append(record)
                break
            else:
                pass
    
    for record in recordsToRemove:
        data.remove(record)
    
    training = data[:(len(data)//10)*6]
    validation = data[(len(data)//10)*6:(len(data)//10)*8]
    test = data[(len(data)//10)*8:]

    return training, validation, test
    
def FindMinMax(data): 
    #Function to find the min and max of each attribute in the data set (later used to standardise the data)
    minMax = [[9999999999,-9999999999] for i in range(len(data[0]))]

    for record in data:
        for i in range(len(record)):
            value = record[i]
            if value < minMax[i][0]:
                minMax[i][0] = value
            if value > minMax[i][1]:
                minMax[i][1] = value

    return minMax

def StandardiseDatasets(dataSet, minMax): 
    #Function to standardise each attribute in the data using their min/max 
    for i in range(len(dataSet)):
        for j in range(len(dataSet[i])):
            dataSet[i][j] = 0.8*((dataSet[i][j] - minMax[j][0])/(minMax[j][1] - minMax[j][0])) + 0.1
   
    training = dataSet[:(len(dataSet)//10)*6]
    validation = dataSet[(len(dataSet)//10)*6:(len(dataSet)//10)*8]
    test = dataSet[(len(dataSet)//10)*8:]

    return training, validation, test, minMax


def DestandardiseResult(datapoint, minMax): 
    #Function to destandardise the result passed to it using the min/max
    datapoint = (((datapoint - 0.1) / 0.8) * (minMax[len(minMax)-1][1] - minMax[len(minMax)-1][0])) + minMax[len(minMax)-1][0]
    return datapoint



def GenerateDatasets(filename): 
    #Function to generate a fresh set of formatted, standardised, cleansed data (training, validation, testing) and save to a JSON for later use
    training, validation, test= ReadData(filename)

    meanSet = CalcMean(training+validation+test)
    SDSet = CalcSD(training+validation+test, meanSet)

    training, validation, test= removeOutliers(training+validation+test, meanSet, SDSet)


    minMax = FindMinMax(training + validation)
    training, validation, test, minMax = StandardiseDatasets(training + validation + test, minMax)


    jsondict = {"training" : training, "validation" : validation, "testing" : test, "minMax" : minMax}

    with open("dataset.json", 'w') as f:
        f.write(json.dumps(jsondict, indent=4))
    
    return training, validation, test, minMax

def ReadDatasetJSON(filename):
    #Function to parse the JSON File and return the stored dataset (plus the min/max used to later destandardise)
    with open(filename, 'r') as f:
        formatted = json.load(f)
    return formatted["training"], formatted["validation"], formatted["testing"], formatted["minMax"]

