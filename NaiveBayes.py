import pandas as pd
import numpy as np
import sys

def getFiletoDF(filename):
    DF = pd.read_csv(filename)
    return DF

def confusionMatrix(true, pred):
    K = len(np.unique(true)) 
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result

def multiplyList(myList) : 
      
    # Multiply elements one by one 
    result = 1
    for x in myList: 
         result = result * x  
    return result

def createModel(trainFile, modelFile):
    trainDF = getFiletoDF(trainFile)
    trainColumnsN = trainDF.shape[1]
    columns = trainDF.columns
    probs=[]
    
    for colName in columns[:-1]:
        subDFA = trainDF.loc[trainDF["class"] == 1]
        nYes = subDFA.shape[0]
        subDFB = trainDF.loc[trainDF["class"] == 0]
        nNo = subDFB.shape[0]
        
        unq = trainDF[colName].unique()
        unqPs = []
        for v in unq:
            subsubDFA = subDFA.loc[subDFA[colName] == v]
            subsubDFB = subDFB.loc[subDFB[colName] == v]
            
            pVyCy = subsubDFA.shape[0]/nYes
            pVnCy = 1-pVyCy
            pVyCn = subsubDFB.shape[0]/nNo
            pVnCn = 1-pVyCn
            
            unqPs.append("Probabilty "+str(colName)+" = "+str(v)+", given Class = Yes : "+ str(pVyCy)+"\n")
            unqPs.append("Probabilty "+str(colName)+" = NOT "+str(v)+", given Class = Yes : "+ str(pVnCy)+"\n")
            unqPs.append("Probabilty "+str(colName)+" = "+str(v)+", given Class = No : "+ str(pVyCn)+"\n")
            unqPs.append("Probabilty "+str(colName)+" = NOT "+str(v)+", given Class = No : "+ str(pVnCn)+"\n\n")
        
        probs.append(unqPs)
    
        with open(modelFile, 'w') as file:
            file.write("Naive Bayesian Model: \n\n")
            file.write("Nodes: \n")
            for i in range(0, trainColumnsN-1):
                file.write(str(i+1)+". "+str(trainDF.columns[i])+".\n")
            file.write("\nEdges: \n")
            for i in range(0, trainColumnsN-1):
                file.write(str(i+1)+". "+str(trainDF.columns[i])+" TO "+str(trainDF.columns[-1])+".\n")
            file.write("\nProbabilities: \n")
            for list in probs:
                for string in list:
                    file.write(string)

def naiveBayes(trainFile, testFile):
    trainDF = getFiletoDF(trainFile)
    testDF = getFiletoDF(testFile)
    
    testN = testDF.shape[0]

    predictions = []
    
    #for each row in test df
    for n in range(0, testN):

        row = testDF.iloc[n]
        subDFA = trainDF.loc[trainDF["class"] == 1]
        subDFB = trainDF.loc[trainDF["class"] == 0]
        NsubDFA = subDFA.shape[0]
        NsubDFB = subDFB.shape[0]
        NCountA = 0
        NCountB = 0
        
        
        ConditionalPsA = []
        ConditionalPsB = []

        for featureX in row[:-1].index.values:
            subSubDFA = subDFA.loc[subDFA[featureX] == row[featureX]]
            subSubDFB = subDFB.loc[subDFB[featureX] == row[featureX]]
            NcountA = subSubDFA.shape[0]
            NcountB = subSubDFB.shape[0]
        
            ConditionalPsA.append(NcountA/NsubDFA)
            ConditionalPsB.append(NcountB/NsubDFB)

        A = multiplyList(ConditionalPsA)
        
        B = multiplyList(ConditionalPsB)
        
        if(B!=0):
            if(A/B >=1 ):
                predictions.append(1)
            else:
                predictions.append(0)
        else:
            predictions.append(1)
    
    return (predictions, testDF["class"])

def main(trainFile, testFile, modelFile, resultFile):
	trainDF = getFiletoDF(trainFile)
	testDF = getFiletoDF(testFile)

	createModel(trainFile, modelFile)
	
	(pred, true) = naiveBayes(trainFile, testFile)
	
	[[TP, FP], [FN, TN]] = confusionMatrix(true, pred)

	with open(resultFile, 'w') as file:

		file.write("\nConfusion matrix: \n")
		file.write("True Postitives = " + str(TP) + "\n")
		file.write("False Postitives = " + str(FP) + "\n")
		file.write("True Negatives = " + str(TN) + "\n")
		file.write("Fasle Negatives = " + str(FN) + "\n")
		file.write("\nPredicted and Actual values: \n")

		n = len(pred)
		for i in range(0,n):
			file.write("Row "+str(i)+": Predicted = "+str(pred[i])+", and Actual = "+str(true[i])+"\n")

	print("Done!")
		
		
if __name__ == "__main__":

    trainFile = str(sys.argv[1])
    testFile = str(sys.argv[2])
    modelFile = str(sys.argv[3])
    resultFile = str(sys.argv[4])

    main(trainFile, testFile, modelFile, resultFile)

