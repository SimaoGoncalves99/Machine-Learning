#Load libraries 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.model_selection  as sk
import sklearn.metrics

#Load Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


#Plot
plt.imshow(255- x_train[0,:,:], cmap = 'gray')
plt.show()

#Divide 
x_train = x_train/255
x_test = x_test/255

#Convert y to one hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=None, dtype="float32") #None converte para o valor máximo + 1 que serão 10 classes
y_test = tf.keras.utils.to_categorical(y_test, num_classes=None, dtype="float32")

#Split into training and validation
x_train, x_validation, y_train, y_validation = sk.train_test_split(x_train, y_train, train_size=0.8)

#*************************************************Parte 2*******************************************************************************

#Add the flattened layer
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28,28)))

#Add the hidden layers with 32 and 64 neurons 
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))

#Add the softmax layer 
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

#Add an extra dimension 
x_test= np.expand_dims(x_test, axis = 3)   
x_train= np.expand_dims(x_train, axis = 3)   
x_validation= np.expand_dims(x_validation, axis = 3) 

#Check if everything is ok
model.summary() 

#Create an early stop 
es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

#Fit the MLP to the data
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm = 1 )
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

history = model.fit(x = x_train, y = y_train, epochs = 200, callbacks = es, validation_data=(x_validation, y_validation), verbose = 0, batch_size=200)

#Plot the data 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss with early stop',fontsize=14)
plt.xlabel("Epochs",fontsize=12)
plt.ylabel("Loss",fontsize=12)
plt.legend(['training','validation'])
plt.show()

#Performance Evaluation
predict = model.predict(x_test)

#argmax para comparar se o resultado e a previsao correspondem a mesma peça de roupa (mesmo indice)
accuracy = sklearn.metrics.accuracy_score(np.argmax(y_test, axis=1), np.argmax(predict, axis = 1)) 
print("Accuracy with early stop: ", accuracy)

c_mat = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predict, axis = 1))
print("Confusion Matrix with early stop: \n", c_mat)


#Second model
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Flatten(input_shape = (28,28,1)))
model2.add(tf.keras.layers.Dense(32, activation = 'relu'))
model2.add(tf.keras.layers.Dense(64, activation = 'relu'))
model2.add(tf.keras.layers.Dense(10, activation = 'softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm = 1 )
model2.compile(loss='categorical_crossentropy', optimizer=optimizer)

history2 = model2.fit(x = x_train, y = y_train, epochs = 200, validation_data=(x_validation, y_validation), verbose = 0,  batch_size=200)

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Loss without early stop',fontsize=14)
plt.xlabel("Epochs",fontsize=12)
plt.ylabel("Loss",fontsize=12)
plt.legend(['training','validation'])
plt.show()


#Performance Evaluation
predict_2 = model2.predict(x_test)

#argmax para comparar se o resultado e a previsao correspondem a mesma peça de roupa (mesmo indice)
accuracy = sklearn.metrics.accuracy_score(np.argmax(y_test, axis=1), np.argmax(predict_2, axis = 1)) 
print("Accuracy with early stop: ", accuracy)

c_mat = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predict_2, axis = 1))
print("Confusion Matrix with early stop: \n", c_mat)