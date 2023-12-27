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
Exercise: 2.3 

"""

#Load training data 
X = np.load('data1_x.npy')
Y = np.load('data1_y.npy')


C = design_mat(X, 1)
C_T = np.transpose(C)

#Calculate the pseudo inverse 
pInv =np.matmul(np.linalg.inv(np.matmul(C_T, C)),C_T) 

#Estimate the function coefficients 
Beta = np.matmul(pInv, Y)

#plot the results
Y_hat = np.matmul(C,Beta)
plt.plot(X, Y_hat)
plt.scatter(X,Y, c="r", marker = "o")
plt.ylabel("Y")
plt.xlabel("X")
plt.show()
#Compute the sum of squared errors
e = Y - Y_hat
SSE = np.matmul(np.transpose(e),e)
print('\nExercise 2.3 \n')
print('Coefficients (Beta):' , Beta)
print('\nSSE:' , SSE)

"""
Exercise: 2.4

"""
X = np.load('data2_x.npy')
Y = np.load('data2_y.npy')

C = design_mat(X, 2)
C_T = np.transpose(C)

#Calculate the pseudo inverse 
pInv =np.matmul(np.linalg.inv(np.matmul(C_T, C)),C_T) 

#Estimate the function coefficients 
Beta = np.matmul(pInv, Y)

#plot the results
Y_hat = np.matmul(C,Beta)
plt.plot(X, Y_hat)
plt.scatter(X,Y, c="r", marker = "o")
plt.ylabel("Y")
plt.xlabel("X")
plt.show()
#Compute the sum of squared errors
e = Y - Y_hat
SSE = np.matmul(np.transpose(e),e)

print('\nExercise 2.4 \n')
print('Coefficients (Beta):' , Beta)
print('\nSSE:' , SSE)