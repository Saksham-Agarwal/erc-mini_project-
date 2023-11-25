import numpy as np
import csv

test_Data = []
test_Ans = []
the_test_q = []
the_test_A = []
with open("MNIST_CSV/mnist_train.csv", "r") as f:
    csv_r = csv.reader(f)
    for n in csv_r:
        y = [0,0,0,0,0,0,0,0,0,0]
        y[int(n[0]) - 1] = 1
        test_Ans.append(y)
        test_Data.append(n[1:])
with open("MNIST_CSV/mnist_test.csv", "r") as f:
    csv_r = csv.reader(f)
    for n in csv_r:
        y = [0,0,0,0,0,0,0,0,0,0]
        y[int(n[0]) - 1] = 1
        the_test_A.append(y)
        the_test_q.append(n[1:])

the_test_q = np.array(the_test_q[0:10], dtype= float)
the_test_A= np.array(the_test_A[0:10], dtype= float)
test_Ans = np.array(test_Ans[0:1000], dtype = float)
test_Data = np.array(test_Data[0:1000], dtype = float)
np.random.seed(0)

w1 = np.random.rand(28,784)
w2 = np.random.rand(10,28)
b1= np.random.rand(28,1)
b2 = np.random.rand(10,1)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def softmax(z):
    A = np.exp(z)/sum(np.exp(z))
    return A


def forward_prop(X,w1,w2,b1,b2):
    z1 = w1.dot(X.T) + b1
    a1 = sigmoid(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1,a1,z2,a2

def der_sigmoid(a):
    return a * (1  -a)

def backward_prop(w1,w2,a1,a2,z1,z2, Y,X):
    m = Y.size
    dz2 = a2 - Y.T
    dw2 = 1/m * dz2.dot(a1.T)
    db2 = 1/m *  np.sum(dz2)
    dz1 = w2.T.dot(dz2) * der_sigmoid(z1)
    dw1 = 1/m * dz1.dot(X)
    db1 = 1/m * np.sum(dz1)

    return dw1, dw2, db1,db2

iterations = 1000
for i in range(iterations):
    if i%20 == 0:
        print("Iteration: ", i)
    z1,a1,z2,a2 = forward_prop(test_Data,w1,w2,b1,b2)
    dw1, dw2, db1,db2 = backward_prop(w1,w2,a1,a2,z1,z2, test_Ans,test_Data)

    w1 = w1 - 0.1 *dw1
    w2 = w2 - 0.1 *dw2
    b1 = b1 - 0.1 *db1
    b2 = b2 - 0.1 *db2



for n in the_test_q:
    z1,a1,z2,a2 = forward_prop(the_test_q,w1,w2,b1,b2) 
    print(a2)
    