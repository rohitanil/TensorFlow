#Import necessary libraries
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.utils

#Load Iris dataset
df= pd.read_csv("C:\\Users\\rohit.a\\Downloads\\Iris.csv")

cols=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
epoch=10000
n_nodes_h1=100
n_nodes_h2=100
n_classes=3


#One hot encoding target variables
for i in range(0,len(df)):
    if df.Species[i]=='Iris-setosa':
        df.Species[i]=np.asarray([1,0,0])
    elif df.Species[i]=='Iris-versicolor':
        df.Species[i]=np.asarray([0,1,0])
    else:
        df.Species[i]=np.asarray([0,0,1])


df = sklearn.utils.shuffle(df)
df = df.reset_index(drop=True)
x1=df[cols]
y1=df.Species


xtrain,xtest,ytrain,ytest=train_test_split(x1,y1,test_size=0.2,random_state=20)

x= tf.placeholder(tf.float32, shape=[None,4])
y_=tf.placeholder(tf.float32, shape=[None,3])

#Initializing weights and biases of hidden layers and output layer
hidden_layer1={'weights':tf.Variable(tf.random_normal([4,100])),'bias':tf.Variable(tf.random_normal([100]))}
hidden_layer2={'weights':tf.Variable(tf.random_normal([100,100])),'bias':tf.Variable(tf.random_normal([100]))}
output_layer= {'weights':tf.Variable(tf.random_normal([100,3])),'bias':tf.Variable(tf.random_normal([3]))}

#Activation function of hidden layers and output layer
l1=tf.add(tf.matmul(x,hidden_layer1['weights']),hidden_layer1['bias'])
l1=tf.nn.softmax(l1)

l2=tf.add(tf.matmul(l1,hidden_layer2['weights']),hidden_layer2['bias'])
l2=tf.nn.softmax(l2)

l3=tf.add(tf.matmul(l2,output_layer['weights']),output_layer['bias'])
y=tf.nn.softmax(l3)

#cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#optimiser
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#calculating accuracy of our model 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
#initialising variables
init = tf.global_variables_initializer()
sess.run(init)

for step in range(0,epoch):
   _,c=sess.run([train_step,cross_entropy], feed_dict={x: xtrain, y_:[t for t in ytrain.as_matrix()]})
   if step% 500==0 :
       print("Loss at step %d: %f" %(step,c))

print("Accuracy",sess.run(accuracy,feed_dict={x: xtest, y_:[t for t in ytest.as_matrix()]}))
