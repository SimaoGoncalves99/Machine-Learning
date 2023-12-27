#Load libraries 
import numpy as np
import matplotlib.pyplot as plt

"""
Function design_mat()

Receives the training data "X" and the degree of the polynomial we want to fit. 

"""
def design_mat(X,P):
    l = len(X)
    mat = np.ones((l,P + 1 ))
    for i in range(P):
        for j in range(l):
            mat[j][i+1]= X[j]**(i+1)
    
    return mat
"""
Exercise: 2.1.3 

"""

#Load training data 
X = np.load('data1_x.npy')
Y = np.load('data1_y.npy')

#Convert the training data into a design matrix 
C = design_mat(X, 1)

#Calculate the pseudo inverse 
pInv =np.linalg.pinv(C)

#Estimate the function coefficients 
Beta = np.matmul(pInv, Y)

#Get the estimated fit
Y_est = np.matmul(C,Beta)

#plot the results
plt.plot(X, Y_est)
plt.scatter(X,Y, c="r", marker = "o")
plt.title('Polynomial fit for P=1',fontsize=14)
plt.ylabel("Y",fontsize=12)
plt.xlabel("X",fontsize=12)
plt.show()

#Compute the sum of squared errors
e = Y - Y_est
SSE = np.matmul(np.transpose(e),e)
print('\nExercise 2.1.3 \n')
print('Coefficients (Beta):' , Beta)
print('\nSSE:' , SSE)

"""
Exercise: 2.1.4

"""
#Load training data 
X = np.load('data2_x.npy')
Y = np.load('data2_y.npy')

#Convert the training data into a design matrix 
C = design_mat(X, 2)

#Calculate the pseudo inverse 
pInv = np.linalg.pinv(C); 

#Estimate the function coefficients 
Beta = np.matmul(pInv, Y)

#Get the estimated fit
Y_est = np.matmul(C,Beta)

#Sort the results
new_X, new_Y = zip(*sorted(zip(X, Y_est)))

#plot the results
plt.plot(new_X, new_Y)
plt.scatter(X,Y, c="r", marker = "o")
plt.title('Polynomial fit for P=2',fontsize=14)
plt.ylabel("Y",fontsize=12)
plt.xlabel("X",fontsize=12)
plt.show()

#Compute the sum of squared errors
e = Y - Y_est
SSE = np.matmul(np.transpose(e),e)

print('\nExercise 2.1.4 \n')
print('Coefficients (Beta):' , Beta)
print('\nSSE:' , SSE)

"""
Exercise: 2.1.5

"""
#Load training data 
X = np.load('data2a_x.npy')
Y = np.load('data2a_y.npy')

#Convert the training data into a design matrix 
C = design_mat(X, 2)

#Calculate the pseudo inverse 
pInv = np.linalg.pinv(C); 

#Estimate the function coefficients 
Beta = np.matmul(pInv, Y)

#Get the estimated fit
Y_est = np.matmul(C,Beta)

#Sort the results
new_X, new_Y = zip(*sorted(zip(X, Y_est)))

#plot the results
plt.plot(new_X, new_Y)
plt.scatter(X,Y, c="r", marker = "o")
plt.title('Polynomial fit for P=2 (with outliers)',fontsize=14)
plt.ylabel("Y",fontsize=12)
plt.xlabel("X",fontsize=12)
plt.show()

#Compute the sum of squared errors
e = Y - Y_est
SSE = np.matmul(np.transpose(e),e)
print('\nExercise 2.1.5 \n')
print('Coefficients (Beta):' , Beta)
print('\nSSE:' , SSE)

#Compute the SSE without outliers
for k in range(2):
    a = np.argmax(Y)
    Y[a] = 0
    e[a] = 0

SSE = np.matmul(np.transpose(e),e)
print('\nSSE without outliers:' , SSE)

"""
Part 2
""" 
#Load libraries 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as sk

#Load training data 
X = np.load('data3_x.npy')
Y = np.load('data3_y.npy')

#Compute the alphas to be tested and create vectors to store results
alpha = np.arange(0.001,10,0.01) 
ridgeRes = np.zeros((1000,3))  #1000 intervalos e polinómio de grau 2
lassoRes = np.zeros((1000,3))

#Find the Ridge and Lasso coefficients for each value of alpha
for i in range(1000):
    aux = sk.Ridge(alpha = alpha[i], max_iter = 10000) 
    aux.fit(X,Y) #fit to training data
    ridgeRes[i] = aux.coef_
    aux = sk.Lasso(alpha = alpha[i], max_iter = 10000) 
    aux.fit(X,Y)
    lassoRes[i] = aux.coef_

#Calculate the coefficients through LS method (alpha = 0)   
ridge_aux = sk.Ridge(alpha = 0, max_iter = 10000) 
ridge_aux.fit(X,Y)
ridgeLS = ridge_aux.coef_
lasso_aux = sk.Lasso(alpha = 0, max_iter = 10000) 
lasso_aux.fit(X,Y)
lassoLS = lasso_aux.coef_

#plot the results

plt.plot(alpha, ridgeRes)
plt.axhline(y = ridgeLS[0][0], color = 'm', ls = '--')
plt.axhline(y = ridgeLS[0][1], color = 'c', ls = '--')
plt.axhline(y = ridgeLS[0][2], color = 'y', ls = '--')
plt.legend(['\u03b2','\u03b2','\u03b2','\u03b2','\u03b2','\u03b2'])
plt.title('Ridge and LS coefficients evolution with \u03B1',fontsize=14)
plt.xscale("log")
plt.xlabel("\u03B1",fontsize=12)
plt.ylabel("Ridge and LS coefficients",fontsize=12)
plt.show()

plt.plot(alpha, lassoRes)
plt.axhline(y = lassoLS[0], color = 'm', ls = '--')
plt.axhline(y = lassoLS[1], color = 'c', ls = '--')
plt.axhline(y = lassoLS[2], color = 'y', ls = '--')
plt.legend(['\u03b2','\u03b2','\u03b2','\u03b2','\u03b2','\u03b2'])
plt.title('Lasso and LS coefficients evolution with \u03B1',fontsize=14)
plt.xscale("log")
plt.xlabel("\u03B1",fontsize=12)
plt.ylabel("Lasso and LS coefficients",fontsize=12)
plt.show()

#The Y vector was changed, needs to return to normal
X = np.load('data3_x.npy')
Y = np.load('data3_y.npy')

#Lasso regression for an adequate value of alpha
alp = alpha[7] #We choose an alpha where the second feature is irrelevant and the other were better estimated
lasso = sk.Lasso(alpha = alp, max_iter = 10000)
lasso.fit(X,Y)
lassoCoefs = lasso.coef_
pred = lasso.predict(X)
SSE_lasso = 0
#Lasso predict for alpha = 0
pred_0 = lasso_aux.predict(X)
SSE_LS = 0
#Compute the sum of squared errors 
for i in range(50):
    SSE_lasso += (Y[i]-pred[i])**2
    SSE_LS += (Y[i]-pred_0[i])**2

#print the results
print('\nSSE Lasso:',SSE_lasso)
print('\nSSE LS:', SSE_LS)


#Plot y in comparison with the LS fit
plt.plot(pred_0)
plt.plot(Y)
plt.title('LS fit (\u03B1 = 0)',fontsize=14)
plt.xlabel("Pattern Number",fontsize=12)
plt.ylabel("y_Data",fontsize=12)
plt.legend(['Y','LS regression'])
plt.show()

#Plot y in comparison with the Lasso fit for the chosen alpha
plt.plot(pred)
plt.plot(Y)
plt.title('Lasso fit for \u03B1 = %f' %alp,fontsize=14)
plt.xlabel("Pattern Number",fontsize=12)
plt.ylabel("Y",fontsize=12)
plt.legend(['Y','Lasso regression'])
plt.show()


