'''
    Cat or not?

    @python 3.83 ('base':conda)
    @author : Qinghao Zhang
    @Time : 2022/4/23

'''

import numpy as np
import h5py
import matplotlib.pyplot as plt

from lr_utils import load_dataset

"""准备工作: 调调矩阵维度"""

train_X_orig, train_Y, test_X_orig, test_Y, classes = load_dataset()
print ("Shape of train_set_x_orig: " + str(train_X_orig.shape))     #   打印训练集输入的形状 
print ("Shape of train_set_y: " + str(train_Y.shape))   #   打印训练集输出的形状
print ("Shape of test_set_x_orig: " + str(test_X_orig.shape))   #   打印测试集输入的形状
print ("Shape of test_set_y: " + str(test_Y.shape))    #    测试集输出的形状

#   将训练集的维度降低并转制
train_set_X_flatten = train_X_orig.reshape(train_X_orig.shape[0],-1).T
#   将测试集的维度降低并转制
test_set_X_flatten = test_X_orig.reshape(test_X_orig.shape[0],-1).T
print ("Shape of train_set_X_flatten: " + str(train_set_X_flatten.shape))
print ("Value of test_set_X_flatten: " + str(test_set_X_flatten))

#   将64*64像素值转化为0-1范围数， 因为imshow()显示图像时对double型认为01范围ie

train_set_x = 1.0 * train_set_X_flatten / 255
test_set_x = 1.0 * test_set_X_flatten / 255
print ("Shape of train_set_x: " + str(train_set_x.shape))
print ("Value of test_set_x: " + str(test_set_x)) 

"""     搭建深度学习框架      """

#   实现sigmod
def sigmoid(z):
    s = 1. / (1 + np.exp(-z))
    return s

#   实现 初始化权值
def initialize(dim):
    '''设置 w, b
       其中 w 的维度为 dim , w 为零向量
       初始化 b = 0
    '''
    w = np.zeros(shape = (dim,1))
    b = 0
    return (w,b)

#   实现 正反向传播
def propagate(w, b, X, Y):
    m = X.shape[1] # m 表示样本的列数，即样本容量 m 个

    #正向传播
    A = sigmoid(np.dot(w.T,X) + b)  #激活值
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A))) #成本函数

    #反向传播
    dw = (1 / m) * np.dot(X, (A-Y).T)
    db = (1 / m) * np.sum(A-Y)

    #创建字典
    grads = {
                "dw": dw,
                "db": db
    }
    return (grads , cost)


#   实现 更新 w, b
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):
    
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 10 == 0:
            costs.append(cost)
        if print_cost and i%10 == 0:
            print ("迭代次数 %i. 误差值: %f"%(i,cost))
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return (params, grads, costs)

#   实现 预测
def predict(w, b, X):
    m = X.shape[1] # m个样本
    Y_hat = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T,X) + b)
    for i in range(A.shape[1]):
        Y_hat[0,i] = 1 if A[0,i] > 0.5 else 0
    return Y_hat

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w,b = initialize(X_train.shape[0])
    
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)

    w,b=parameters["w"],parameters["b"]

    Y_hat_test = predict(w,b,X_test)
    Y_hat_train = predict(w,b,X_train)

    print("训练集的准确性: " + format(100 - np.mean(np.abs(Y_hat_train-Y_train))*100) + "%")
    print("测试集的准确性: " + format(100 - np.mean(np.abs(Y_hat_test - Y_test))*100) + "%")

    d = {
        "costs":costs,
        "Y_predicition_test" : Y_hat_test,
        "Y_predicition_train" : Y_hat_train,
        "w" : w,
        "b" : b,
        "learning_rate" : learning_rate,
        'num_interations' : num_iterations
    }
    return d
d = model(train_set_x,train_Y,test_set_x,test_Y,num_iterations=5000, learning_rate=0.02,print_cost=True)

#绘制图

costs = np.squeeze(d['costs'])

plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hunderds)')
plt.title("Learning rate = " + str(d["learning_rate"]))
plt.show()