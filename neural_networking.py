
import numpy as np
import csv


#some_variables i am determining before for my sake

y = [0,0,0,0,0,0,0,0,0,0]
# no of neurons in hidden layer 1 and 2
n_n = 28
m_n = 28

#no of neurons in input
n_i = 784


#no of neurons in output
n_o = 10

lr = 0.1

np.random.seed(420)

w1 = np.random.rand(n_n, n_i)
w2 = np.random.rand(n_n, n_n)
w3 = np.random.rand(n_o, n_n)
b1 = np.random.rand(n_n)
b2 =  np.random.rand(n_n)


def sigmoid(z):
    z = 1/(1 + np.exp(-z))
    return z

def forward_prop(pixels_, w1,w2,w3,b1,b2):
    z1 = np.dot(w1, pixels_.T) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(w3, a2)
    a3 = sigmoid(z3)
    
    return z1,a1,z2,a2,z3,a3

def back_prop(a1,a2,a3,y,m_n,x):
    
    dz3 = 2 *(a3 - y)
    dz3 = dz3.reshape(10,1)
    a2 = a2.reshape(1,28) 

    dw3 = np.dot(dz3, a2)/m_n
    dz2 = np.dot(dw3.T, dz3) * a2 * (1 - a2)
    db2  =dz3 * a2 * (1 - a2)
    a1 =a1.reshape(1,28)
    dw2 = np.dot(dz2, a1.T)/m_n
    dz1 = np.dot(dw2.T, dz2) * a1 * (1 - a1)
    x = x.reshape(1,784)
    dz1 = dz1.reshape(28,1)
    dw1 = np.dot(dz1,x)/m_n
    dw1 = np.reshape(dw1, w1.shape)
    dw2 = np.reshape(dw2, w2.shape)
    dw3 = np.reshape(dw3, w3.shape)
    
    db1 = dz2 * a1 * (1 - a1)
    db1 = np.reshape(db1, b1.shape)
    db2 = np.reshape(db2, b2.shape)

    return dz3, dw3, dw2, dz2, db2, dw2, dz1, db1, dw1 

with open("MNIST_CSV/mnist_train.csv") as f:
    csv_r = csv.reader(f)
    x = 1

    for i in csv_r:
        print(x)
        x+= 1
        y_ = int(i[0])
        y[y_ - 1] = 1
        y = np.array(y)
        x = np.array(i[1::],dtype = int)
        z1,a1,z2,a2,z3,a3 =forward_prop(x, w1,w2,w3,b1,b2)

        dz3,dw3,dz2,db2,dw2,dz1,db1, dw1 = back_prop(a1,a2,a3,y,m_n,x)

        w3 = w3 - lr * dw3
        w2 = w2 - lr* dw2
        w1 = w1 - lr* dw1
        b2 = b2 - lr* db2
        b1 = b1 - lr* db1

# only this much part showed error and i was not able to fix it so i left the readin and comparing part after this





        
