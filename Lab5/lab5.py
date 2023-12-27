#Load libraries 
import numpy as np
import tensorflow as tf
import sklearn.model_selection  as sk
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, tree
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import seaborn as sns



######################## CLASSIFICATION TASK ##################################

#Load training and test data
(xtrain, ytrain) = (np.load('classification_data/Cancer_Xtrain.npy'),np.load('classification_data/Cancer_ytrain.npy'))
(xtest, ytest) = (np.load('classification_data/Cancer_Xtest.npy'),np.load('classification_data/Cancer_ytest.npy'))


#Prescale the data  (Comentar para testes sem pré-processamento)
scaler = sklearn.preprocessing.StandardScaler().fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

#Visualização da data
visualize = pd.DataFrame(data=np.concatenate((xtrain,ytrain),axis=1))
sns.pairplot(visualize, hue = 9) #Comentando o Hue pode ser obtida a data em forma de histograma, no entanto será
                                 #adicionada a concatenação das classes

#SUPPORT VECTOR MACHINE CLASSIFIER (RBF Kernel)
ytrain = np.ravel(ytrain, 'C')

#Define the range for C and gamma
C_range = np.logspace(-3, 6, 10)
gamma_range = np.logspace(-3, 3, 7)
#Parameter grid
param_grid = dict(gamma = gamma_range, C = C_range)
#K-Fold cross validation
kf = sklearn.model_selection.KFold(n_splits=4,random_state=None, shuffle=False); 
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=kf, n_jobs=-1)
grid.fit(xtrain, ytrain)

print("The best parameters are %s with a score of %0.2f\n"
      % (grid.best_params_, grid.best_score_))


#Training the SVM (RBF Kernel)
SVMclassifier = SVC(kernel='rbf',C = grid.best_estimator_.C, gamma = grid.best_estimator_.gamma)
SVMclassifier.fit(xtrain,ytrain)

#Predictions
pred = SVMclassifier.predict(xtest)

#Classifier accuracy on the test data
SVM_rbf_acc = sklearn.metrics.accuracy_score(ytest, pred)
SVM_rbf_fmeasure = sklearn.metrics.f1_score(ytest, pred)
SVM_rbf_error = (1-SVM_rbf_acc)*100
SVM_rbf_confusionM = sklearn.metrics.confusion_matrix(ytest,pred);
SVM_rbf_precision = sklearn.metrics.precision_score(ytest,pred);
SVM_rbf_sensivity = sklearn.metrics.balanced_accuracy_score(ytest,pred);

print("(SVM (rbf kernel) classifier) Accuracy: \n", SVM_rbf_acc)
print("(SVM (rbf kernel) classifier) F Measure: \n", SVM_rbf_fmeasure)
print("(SVM (rbf kernel) classifier) Confusion Matrix : \n", SVM_rbf_confusionM)
print("(SVM (rbf kernel) classifier) Precision: \n", SVM_rbf_precision)
print("(SVM (rbf kernel) classifier) Sensivity: \n", SVM_rbf_sensivity)

#SUPPORT VECTOR MACHINE CLASSIFIER (Polynomial Kernel)

#Training the SVM (Linear Kernel)
SVMclassifier = SVC(kernel='linear',C=1)
SVMclassifier.fit(xtrain,ytrain)

#Predictions
vect = np.ones(100)
#Training the SVM (polynomial Kernel) (p=3)
#for i in range(100):
#    SVMclassifier = SVC(kernel='poly',C =0.001, gamma = 0.001, degree = i)#Coef0 = 0 by standard
#    SVMclassifier.fit(xtrain,ytrain)
#    pred = SVMclassifier.predict(xtest)
#    vect[i] = sklearn.metrics.accuracy_score(ytest, pred)
#degree = np.argmax(vect)
#Predictions
SVMclassifier = SVC(kernel='poly',C =0.001, gamma = 0.001, degree = 3)
SVMclassifier.fit(xtrain,ytrain)
pred = SVMclassifier.predict(xtest)

#Classifier accuracy on the test data
SVM_poly_acc = sklearn.metrics.accuracy_score(ytest, pred)
SVM_poly_fmeasure = sklearn.metrics.f1_score(ytest, pred)
SVM_poly_error = (1-SVM_poly_acc)*100
SVM_poly_confusionM = sklearn.metrics.confusion_matrix(ytest,pred);
SVM_poly_precision = sklearn.metrics.precision_score(ytest,pred);
SVM_poly_sensivity = sklearn.metrics.balanced_accuracy_score(ytest,pred);

print("\n(SVM (poly kernel) classifier) Accuracy: \n", SVM_poly_acc)
print("(SVM (poly kernel ) classifier) F Measure: \n", SVM_poly_fmeasure)
print("(SVM (poly kernel) classifier) Confusion Matrix : \n", SVM_poly_confusionM)
print("(SVM (poly kernel ) classifier) Precision: \n", SVM_poly_precision)
print("(SVM (poly kernel) classifier) Sensivity: \n", SVM_poly_sensivity)

##  
##NAIVE BAYES CLASSIFIER

#The features are conditionally independent

#Load training and test data
(xtrain, ytrain) = (np.load('classification_data/Cancer_Xtrain.npy'),np.load('classification_data/Cancer_ytrain.npy'))
(xtest, ytest) = (np.load('classification_data/Cancer_Xtest.npy'),np.load('classification_data/Cancer_ytest.npy'))

naivebayes = GaussianNB()
ytrain = np.ravel(ytrain,'C');
pred = naivebayes.fit(xtrain,ytrain).predict(xtest);

#Classifier accuracy on the test data
NB_acc = sklearn.metrics.accuracy_score(ytest, pred);
NB_fmeasure = sklearn.metrics.f1_score(ytest, pred);
NB_error = (1-NB_acc)*100;
NB_confusionM = confusion_matrix(ytest,pred);
NB_precision = sklearn.metrics.precision_score(ytest,pred);
NB_sensivity = sklearn.metrics.balanced_accuracy_score(ytest,pred);

print("\n(Naive Bayes) Accuracy: \n", NB_acc)
print("((Naive Bayes) classifier) F Measure: \n", NB_fmeasure)
print("((Naive Bayes) classifier) Confusion Matrix : \n", NB_confusionM)
print("((Naive Bayes) classifier) Precision: \n", NB_precision)
print("((Naive Bayes) classifier) Sensivity: \n", NB_sensivity)
    
##  
##DECISION TREE CLASSIFIER
clf = tree.DecisionTreeClassifier()
clf.fit(xtrain,ytrain)
pred = clf.predict(xtest)
plt.figure();
tree.plot_tree( clf)

DT_acc = sklearn.metrics.accuracy_score(ytest,pred)
DT_fmeasure = sklearn.metrics.f1_score(ytest, pred);
DT_error = (1-DT_acc)*100;
DT_confusionM = confusion_matrix(ytest,pred);
DT_precision = sklearn.metrics.precision_score(ytest,pred);
DT_sensivity = sklearn.metrics.balanced_accuracy_score(ytest,pred);

print("\n(Decision Tree) Accuracy: \n", DT_acc)
print("((Decision Tree) classifier) F Measure: \n", DT_fmeasure)
print("((Decision Tree) classifier) Confusion Matrix : \n", DT_confusionM)
print("((Decision Tree) classifier) Precision: \n", DT_precision)
print("((Decision Tree) classifier) Sensivity: \n\n", DT_sensivity)


##  
##DECISION TREE EXTRA CLASSIFIER
clf = tree.ExtraTreeClassifier()
clf.fit(xtrain,ytrain)
pred = clf.predict(xtest)
plt.figure();
tree.plot_tree( clf)

DT_acc = sklearn.metrics.accuracy_score(ytest,pred)
DT_fmeasure = sklearn.metrics.f1_score(ytest, pred);
DT_error = (1-NB_acc)*100;
DT_confusionM = confusion_matrix(ytest,pred);
DT_precision = sklearn.metrics.precision_score(ytest,pred);
DT_sensivity = sklearn.metrics.balanced_accuracy_score(ytest,pred);

print("\n(Decision Tree 2) Accuracy: \n", DT_acc)
print("((Decision Tree 2) classifier) F Measure: \n", DT_fmeasure)
print("((Decision Tree 2) classifier) Confusion Matrix : \n", DT_confusionM)
print("((Decision Tree 2) classifier) Precision: \n", DT_precision)
print("((Decision Tree 2 ) classifier) Sensivity: \n\n", DT_sensivity)



########################### REGRESSION TASK ####################################

#Load training and test data
(xtrain, ytrain) = (np.load('regression_data/Real_Estate_Xtrain.npy'),np.load('regression_data/Real_Estate_ytrain.npy'))
(xtest, ytest) = (np.load('regression_data/Real_Estate_Xtest.npy'),np.load('regression_data/Real_Estate_ytest.npy'))

#Visualização da data
plt.figure()
plt.hist(xtrain[:,1])
plt.title('Histogram of one of the parameters',fontsize=14)

ytrain = np.ravel(ytrain, 'C')

#Standardization of the dataset
(xtrain,ytrain) = (sklearn.preprocessing.scale(xtrain),sklearn.preprocessing.scale(ytrain))
(xtest,ytest) = (sklearn.preprocessing.scale(xtest),sklearn.preprocessing.scale(ytest))


#Split data into training and validation
xtrain, xvalidation, ytrain, yvalidation = sk.train_test_split(xtrain, ytrain, train_size=0.8)

######MLP########

model = tf.keras.Sequential(name='MLP')
#Add the hidden layers with 32 and 64 neurons 
model.add(tf.keras.layers.Dense(32,input_dim = 13, activation = 'relu'))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
#Add the linear layer 
model.add(tf.keras.layers.Dense(1, activation = 'linear'))
#Check if everything is ok
model.summary() 

#Create an early stop 
es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

#Fit the MLP to the data using mean squared error as metric 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm = 1)
model.compile(loss='mse', optimizer=optimizer,metrics=['mean_squared_error'])
history = model.fit(x = xtrain, y = ytrain, epochs = 200, callbacks = es, validation_data=(xvalidation, yvalidation), verbose = 0)
    
#Plot of the progression
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss with early stop',fontsize=14)
plt.xlabel("Epochs",fontsize=12)
plt.ylabel("Loss",fontsize=12)
plt.legend(['training','validation'])
plt.show()


#Performance Evaluation
predict = model.predict(xtest)
mean_sqerror = sklearn.metrics.mean_squared_error(ytest,predict)
max_error = sklearn.metrics.max_error(ytest,predict)


print("\n(MLP) Mean Squared Error: \n",mean_sqerror) 
print("(MLP) Max Error: \n",max_error) 

######Ridge Regression########

#Calculate the most adequated alpha with built in CV for this parameter
regressor = linear_model.RidgeCV(alphas=np.logspace(-7, 7, 20))
model = regressor.fit(xtrain,ytrain)

#Performance Evaluation
predict = regressor.predict(xtest)
mean_sqerror = sklearn.metrics.mean_squared_error(ytest,predict)
max_error = sklearn.metrics.max_error(ytest,predict)


print("\n(Ridge Regression) Mean Squared Error: \n",mean_sqerror) 
print("(Ridge Regression) Max Error: \n",max_error) 

######Lasso Regression########

#Calculate the most adequated alpha with built in CV for this parameter
regressor = linear_model.LassoCV(alphas=np.logspace(-7, 7, 20),cv=5)
model = regressor.fit(xtrain,ytrain)

#Performance Evaluation
predict = regressor.predict(xtest)
mean_sqerror = sklearn.metrics.mean_squared_error(ytest,predict)
max_error = sklearn.metrics.max_error(ytest,predict)

print("\n(Lasso Regression) Mean Squared Error: \n",mean_sqerror) 
print("(Lasso Regression) Max Error: \n",max_error) 




