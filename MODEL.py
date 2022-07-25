import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

def sigmoid(z):
    
    s = 1/(1+np.exp(-z))
    return s


def initialize_zeros(dim):
    
    w = np.zeros(shape=(dim,1))
    b=0
    
    assert(w.shape == (dim,1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return (w,b)


def propagate(w,b,X,Y):
    
    m = X.shape[1]
    
    Z = np.dot(w.T,X)+b
    A = sigmoid(Z)
    cost = -1/m * np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    dZ = A - Y
    dw = 1/m * np.dot(X, dZ.T)
    db = 1/m * np.sum(dZ)
    cost = np.squeeze(cost)
    
    assert(dw.shape == w.shape)
    assert(cost.shape == ())
    assert(db.dtype == float)

    diction = {
        "dw":dw,
        "db":db
        }
    
    return (diction, cost)

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = False):
    
    costs = []
    
    for i in range(num_iterations):
        grads,cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw
        
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
            
        if (print_cost) and (i%100 == 0):
            print("迭代的次数：%i ， 误差值：%f ",(i,cost))
            
    params = {
        "w":w,
        "b":b
        }
    
    grads = {
        "dw":dw,
        "db":db
        }
    
    return params,grads,costs

def predict(w,b,X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
     
    A = sigmoid(np.dot(w.T, X)+b)
     
    for i in range(A.shape[1]):
        Y_prediction[0,i] = 1 if A[0,i]>0.5 else 0
    
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction


def model(X_train , Y_train , X_test , Y_test , num_iterations = 2000 , learning_rate = 0.5 , print_cost = False):
    
    w , b = initialize_zeros(X_train.shape[0])
    
    params,grads,costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
    
    w , b = params["w"] , params["b"]
    
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    
    #打印训练后的准确性
    print("训练集准确性："  , format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100) ,"%")
    print("测试集准确性："  , format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) ,"%")
    
    d = {
            "costs" : costs,
            "Y_prediction_test" : Y_prediction_test,
            "Y_prediciton_train" : Y_prediction_train,
            "w" : w,
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations" : num_iterations }
    
    return d



train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()
m_train = train_set_y.shape[1] #训练集里图片的数量。
m_test = test_set_y.shape[1] #测试集里图片的数量。
num_px = train_set_x_orig.shape[1] #训练、测试集里面的图片的宽度和高度（均为64x64）。

index = 88
plt.imshow(train_set_x_orig[index])
#print("train_set_y=" + str(train_set_y)) #你也可以看一下训练集里面的标签是什么样的。


#X_flatten = X.reshape(X.shape [0]，-1).T ＃X.T是X的转置
#将训练集的维度降低并转置。
train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
#将测试集的维度降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


print("====================测试model====================")     
#这里加载的是真实的数据，请参见上面的代码部分。
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    
    