import pandas as pd
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

def loadData(datafile):
    with open(datafile, 'r', encoding = 'latin1') as csvfile:
        data = pd.read_csv(csvfile)

    print(data.columns.values)
    
    return data

#runKNN function updated to include new variable from problem 2)
def runKNN(dataset, prediction, ignore, neighbors):
    X = dataset.drop(columns=[prediction,ignore])
    Y = dataset[prediction].values
    
    #Split the data into a training and testing set
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.4, random_state=1, stratify = Y)
    #Updated to account for 60/40 split
    
    #Run k-NN algorithm
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    
    #Train Model
    knn.fit(X_train, Y_train)
    
    #Test Model
    score = knn.score(X_test, Y_test)
    Y_prediction = knn.predict(X_test)
    F1 = f1_score(Y_test, Y_prediction, average = 'macro')
    
    print('Predicts ' + prediction + ' with ' + str(score) + ' accuracy')
    print('Chance is:' + str(1.0/len(dataset.groupby(prediction))))
    print('F1 Score: ' + str(F1))
    
    return knn

def classifyPlayer(targetRow, data, model, prediction, ignore):
    X = targetRow.drop(columns=[prediction, ignore])
    
    #Determine 5 closest neighbors
    neighbors = model.kneighbors(X,n_neighbors=5, return_distance=False)
    
    #Print out neighbors data
    for neighbor in neighbors[0]:
        print(data.iloc[neighbor])
        
def runKMeans(dataset,ignore):
    #Create dataset
    X = dataset.drop(columns=ignore)
    
    #number below is arbitrary, may need some tinkering to work
    kmeans = KMeans(n_clusters=5)
    
    #train the model
    kmeans.fit(X)
    
    #Add the predictions to the df
    dataset['cluster'] = pd.Series(kmeans.predict(X), index=dataset.index)
    
    #Print a scatterplot matrix
    scatterMatrix = sns.pairplot(dataset.drop(columns=ignore), hue='cluster', palette='Set2')
    
    scatterMatrix.savefig('kmeanClusters.png')
    
    return kmeans
    
#Monday - Problem 1

nbaData = loadData('nba_2013_clean.csv')
knnModel = runKNN(nbaData,'pos','player',5)
classifyPlayer(nbaData.loc[nbaData['player']=='LeBron James'], nbaData, knnModel, 'pos','player')

kmeansmodel = runKMeans(nbaData, ['pos','player'])
