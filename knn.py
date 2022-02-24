# GRR20173548 Miguel Angelo Neumann Salerno ------------------------------

import argparse, operator
import numpy as np
import multiprocessing as mp


# ------------------------------------------------------------------------

def euclideanDistance (element1, element2): #calcula a distancia

	# subtract the arrays
	dif = (element1 - element2)         #um elemento - outro 
	# sum the results squared
	return np.sum(dif ** 2)              #soma os dois e eleva

# ------------------------------------------------------------------------

def getNeighbors (trainingSet_x, trainingSet_y, testElement, k): #passa os atributos de treino, as labels de treino, os atributos do teste e o valor de k

	# calculates the distance for all elements
	distances = [(label, euclideanDistance(testElement, x)) for x, label in zip(trainingSet_x, trainingSet_y)]  #monta vetor com a label do treino passa p funcao atributos de 
														       #teste, atributos de treino e label                              
	# sort the distances array                                                                                 #de treino
	distances.sort(key = operator.itemgetter(1))

	neighbors = []

	# get the k closest classes
	for i in range(k):
		neighbors.append(distances[i][0])

	return neighbors

# ------------------------------------------------------------------------

def getClass (neighbors):

	votes = {}
	
	for i in range(len(neighbors)):
		prediction = neighbors[i]

		# check if already has votes
		if prediction in votes:
			votes[prediction] += 1
		else:
			votes[prediction] = 1

	# sort the number of votes, descending
	sortedVotes = sorted(votes.items(), key = operator.itemgetter(1), reverse = True)

	return sortedVotes[0][0]

# ------------------------------------------------------------------------

def getAcurracy (testSet_y, predictions):

	correct = 0
	length = len(testSet_y)

	for i in range(length):

		# if predicted correctly
		if testSet_y[i] == predictions[i]:
			correct += 1

	# return accuracy
	return (correct/length)

# ------------------------------------------------------------------------

def getConfusionMatrix (testSet_y, predictions):

	# get the number of classes
	size = len(np.unique(testSet_y))

	# create an empty matrix
	m = np.zeros([size, size], dtype = int)

	for i in range(len(testSet_y)):

		# add one to each class predicted
		m[int(testSet_y[i])][int(predictions[i])] += 1

	return m

# ------------------------------------------------------------------------

def preprocess (dataSet):

	data_x = []
	data_y = []

	for line in dataSet:

		# separate the label
		data_y.append(line[0])

		# parse the data, getting only the values 
		data_x.append([float(value.split(":")[1]) for value in line[1:]])

	return np.array(data_x), np.array(data_y)

# ------------------------------------------------------------------------

def knn (trainingSet_x, trainingSet_y, testElement, k):

	# get the neigbors
	neigh = getNeighbors(trainingSet_x, trainingSet_y, testElement, k) 

	# get the class of the neighbors
	result = getClass(neigh)  

	return result

# ------------------------------------------------------------------------



def main ():

	# parse arguments

	parser = argparse.ArgumentParser()
	parser.add_argument('train', type = argparse.FileType('r'), help = 'train file')
	parser.add_argument('test', type = argparse.FileType('r'), help = 'test file')
	parser.add_argument('k', type = int, help = 'number of neighbors')
	args = parser.parse_args()

	# set variables
	trainFile = getattr(args, 'train')
	testFile = getattr(args, 'test')
	kValue = getattr(args, 'k')

	trainingSet = []
	testSet = []

	# read files
	for line in trainFile:
		trainingSet.append(list(line.split())) #separa as linhas em varios elementos

	for line in testFile:
		testSet.append(list(line.split()))

	# preprocess data files
	trainingSet_x, trainingSet_y = preprocess(trainingSet) #realiza processamento de forma que devolve as labels como vetor y e os atributos como vetor x
	testSet_x, testSet_y = preprocess(testSet)		#realiza processamento de forma que devolve as labels como vetor y e os atributos como vetor x

	predictions = []                                #classificacao

	# multiprocess knn function
	pool = mp.Pool(mp.cpu_count())
	parameters = [(trainingSet_x, trainingSet_y, x, kValue) for x in testSet_x]
	predictions = pool.starmap(knn, parameters)              #passa para o knn os labels de treino, os atributos de treino, varia os atributos e passa o valor de K
	pool.close()                                           #retorna a classificacao
	
	# get acurracy
	acurracy = getAcurracy(testSet_y, np.array(predictions))
	print(acurracy)
	
	# get confusion matrix
	confusionMatrix = getConfusionMatrix(testSet_y, np.array(predictions))
	print(confusionMatrix)


# ------------------------------------------------------------------------

if __name__ == "__main__":
	main()

