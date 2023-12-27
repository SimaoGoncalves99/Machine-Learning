#Load libraries 
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from scipy.stats import norm
from scipy.stats import multivariate_normal
import pandas as pd
import sklearn.naive_bayes as nb
import sklearn.feature_extraction.text as fet

#Load training and test data 
xtrain = np.load('data1_xtrain.npy')
ytrain = np.load('data1_ytrain.npy')
xtest = np.load('data1_xtest.npy')
ytest = np.load('data1_ytest.npy')

#Split data by class
xtrain1 = xtrain[0:50]
xtrain2 = xtrain[50:100]
xtrain3 = xtrain[100:150]
xtest1 = xtest[0:50]
xtest2 = xtest[50:100]
xtest3 = xtest[100:150]

#Scatter of the training data
plt.figure()
plt.ylim(-4,8)
plt.xlim(-4,8)
plt.scatter(xtrain1[:,0],xtrain1[:,1], c="r", marker = "o",label = "Class 1")
plt.scatter(xtrain2[:,0],xtrain2[:,1], c="b", marker = "x",label = "Class 2")
plt.scatter(xtrain3[:,0],xtrain3[:,1], c="g", marker = "^",label = "Class 3")
plt.title('Training Data',fontsize=14)
plt.legend(loc='upper right')

#Scatter of the test data
plt.figure()
plt.ylim(-4,8)
plt.xlim(-4,8)
plt.scatter(xtest1[:,0],xtest1[:,1], c="r", marker = "o", label = "Class 1")
plt.scatter(xtest2[:,0],xtest2[:,1], c="b", marker = "x", label = "Class 2")
plt.scatter(xtest3[:,0],xtest3[:,1], c="g", marker = "^", label = "Class 3")
plt.title('Test Data',fontsize=14)
plt.legend(loc='upper right')

#NAIVE BAYES CLASSIFIER
#The features are conditionally independent
 
#probability of data belong to each class
p_i = 1/3

#Mean of each class
mean01 = np.mean(xtrain1[:,0])
mean02 = np.mean(xtrain2[:,0])
mean03 = np.mean(xtrain3[:,0])
mean11 = np.mean(xtrain1[:,1])
mean12 = np.mean(xtrain2[:,1])
mean13 = np.mean(xtrain3[:,1])
var01 = np.var(xtrain1[:,0])
var02 = np.var(xtrain2[:,0])
var03 = np.var(xtrain3[:,0])
var11 = np.var(xtrain1[:,1])
var12 = np.var(xtrain2[:,1])
var13 = np.var(xtrain3[:,1])

#Gaussian distribution of the data 
density01 = norm.pdf(xtest[:,0],mean01,np.sqrt(var01))
density11 = norm.pdf(xtest[:,1],mean11,np.sqrt(var11))
pw1x= p_i*density01*density11

density02 = norm.pdf(xtest[:,0],mean02,np.sqrt(var02))
density12 = norm.pdf(xtest[:,1],mean12,np.sqrt(var12))
pw2x= p_i*density02*density12

density03 = norm.pdf(xtest[:,0],mean03,np.sqrt(var03))
density13 = norm.pdf(xtest[:,1],mean13,np.sqrt(var13))
pw3x= p_i*density03*density13

#Maximum likelihood estimator
pred = np.linspace(0,0,150)
for i in range(150):
    
    pwx = np.array([pw1x[i], pw2x[i], pw3x[i]])
    pred[i] = np.argmax(pwx) + 1 #The prediction is the most probable class
    
#Classifier accuracy on the test data
acc = sklearn.metrics.accuracy_score(ytest, pred)

#Error in %
error = (1-acc)*100
print("(Naive Bayes) % of errors obtained in the test data: \n", error)

#Scatter plot of the test data classification

samples = np.linspace(1,150,150)
plt.figure()
plt.scatter(samples,pred, c="r", marker = "o", label = "Prediction")
plt.scatter(samples,ytest, c="g", marker = "x", label = "Observation")
plt.title('Test Data Classification for Naive Bayes classifier',fontsize=14)
plt.legend(loc='upper right')
plt.xlabel("Samples",fontsize=12)
plt.ylabel("Class",fontsize=12)

#BAYES CLASSIFIER
#The features are no longer conditionally independent

mean1 = np.array([mean01,mean11])
mean2 = np.array([mean02,mean12])
mean3 = np.array([mean03,mean13])
cov1 = np.cov(xtrain1[:,0],xtrain1[:,1])  
cov2 = np.cov(xtrain2[:,0],xtrain2[:,1])  
cov3 = np.cov(xtrain3[:,0],xtrain3[:,1])  

pw1x = p_i*multivariate_normal.pdf(xtest,mean1,cov1)
pw2x = p_i*multivariate_normal.pdf(xtest,mean2,cov2)
pw3x = p_i*multivariate_normal.pdf(xtest,mean3,cov3)

#Maximum likelihood estimator
pred = np.linspace(0,0,150)
for i in range(150):
    
    pwx = np.array([pw1x[i], pw2x[i], pw3x[i]])
    pred[i] = np.argmax(pwx) + 1 #The prediction is the most probable class
    
#Classifier accuracy on the test data
acc = sklearn.metrics.accuracy_score(ytest, pred)

#Error in %
error = (1-acc)*100
print("(Bayes) % of errors obtained in the test data: \n", error)

#Scatter plot of the test data classification

samples = np.linspace(1,150,150)
plt.figure()
plt.scatter(samples,pred, c="r", marker = "o", label = "Prediction")
plt.scatter(samples,ytest, c="g", marker = "x", label = "Observation")
plt.title('Test Data Classification for Bayes classifier',fontsize=14)
plt.legend(loc='upper right')
plt.xlabel("Samples",fontsize=12)
plt.ylabel("Class",fontsize=12)



#Part 3 

#get data which is separated by tab
en_trigram_cnt = pd.read_csv("en_trigram_count.tsv", sep="\t", header = None, index_col = 0)
es_trigram_cnt = pd.read_csv("es_trigram_count.tsv", sep="\t", header = None, index_col = 0)
pt_trigram_cnt = pd.read_csv("pt_trigram_count.tsv", sep="\t", header = None, index_col = 0)
fr_trigram_cnt = pd.read_csv("fr_trigram_count.tsv", sep="\t", header = None, index_col = 0)

#check the data shape and format 
print("Shape : ", en_trigram_cnt.shape, "\n")
head = pd.DataFrame.head(self = en_trigram_cnt ,n = 5)
print("head : \n", head)

#make the training matrix with the trigram counts
Xtrain = np.zeros((4,pt_trigram_cnt.shape[0]))

Xtrain[0,:] = np.transpose(en_trigram_cnt[2])
Xtrain[1,:] = np.transpose(es_trigram_cnt[2])
Xtrain[2,:] = np.transpose(pt_trigram_cnt[2])
Xtrain[3,:] = np.transpose(fr_trigram_cnt[2])


Ytrain = ['en', 'es','pt','fr']

#Naive Bayes object with laplace smoothing
obj = nb.MultinomialNB(class_prior=(0.25,0.25,0.25,0.25), fit_prior = False)

#Create the Model
model = obj.fit(Xtrain, Ytrain)

#Make predictions with the data and compute the accuracy score
prediction = model.predict(Xtrain)

acc = sklearn.metrics.accuracy_score(Ytrain, prediction)
print("\nAccuracy:", acc)

#Build a sentences matrix
senmat = ['Que fácil es comer peras.', 'Que fácil é comer peras.', 'Today is a great day for sightseeing.', 'Je vais au cinéma demain soir.'
          , 'Ana es inteligente y simpática', 'Tu vais à escola hoje.']

#Build the count vectorizer
cVect = fet.CountVectorizer(vocabulary = en_trigram_cnt[1], ngram_range = (3,3), analyzer = 'char')

#Build the testing data from the sentence matrix
Xtest = cVect.fit_transform(senmat)
prediction2  =  model.predict(Xtest)

#Get the sorted probabilities for each class
prob = model.predict_proba(Xtest)
prob = np.sort(prob, axis =1)

#Compute the Margin 
margin = prob[:,3] - prob[:,2]



print('Predicted Languages: \n', prediction2)
print('Score:\n', prob[:,3])
print('Margin: \n', margin)
