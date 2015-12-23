import numpy as np
import math

def read_data():
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    infile = open('data\\hw2_lssvm_all.dat', 'r')
    index = 1
    for line in infile:
        strs = line.strip().split(' ')
        tmp = []
        for i in xrange(len(strs) - 1):
            tmp.append(float(strs[i]))
        if(index <= 400):
            train_x.append(tmp)
            train_y.append(int(strs[len(strs) - 1]))
        else:
            test_x.append(tmp)
            test_y.append(int(strs[len(strs) - 1]))
        index += 1
    return train_x, train_y, test_x, test_y

def lssvm_train(train_x, train_y, gama, lambda_):
    N = len(train_x)
    K = np.zeros((N, N))
    for m in xrange(N):
        for n in xrange(N):
            norm = np.linalg.norm(np.array(train_x[m]) - np.array(train_x[n]))
            K[m, n] = math.exp(-gama * norm * norm)
    K = np.matrix(K)
    #print K
    
    I = np.zeros((N, N))
    for m in xrange(N):
        I[m, m] = 1.0
    I = np.matrix(I)
    
    beta = np.dot((lambda_ * I + K).I, np.array(train_y).transpose())
    return beta

def lssvm_predict(train_x, x, beta, gama):
    sum = 0.0
    for n in xrange(len(train_x)):
        norm = np.linalg.norm(np.array(train_x[n]) - np.array(x))
        k = math.exp(-gama * norm * norm)
        sum += beta[0, n] * k
    return sum

def lssvm_test(train_x, test_x, test_y, beta, gama):
    err = 0.0
    for n in xrange(len(test_x)):
        h = lssvm_predict(train_x, test_x[n], beta, gama)
        if(h >= 0.0):
            h = int(1)
        else:
            h = int(-1)
        if(h != int(test_y[n])):
            err += 1
    return float(err / len(test_x))

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = read_data()
    
    gama_list = [32,2,0.125]
    lambda_list = [0.001,1,1000]
    
    err_best = 10.0
    for gama in gama_list:
        for lambda_ in lambda_list:
            beta = lssvm_train(train_x, train_y, gama, lambda_)
            err = lssvm_test(train_x, test_x, test_y, beta, gama)
            print 'gama = ' + str(gama)
            print 'lambda_ = ' + str(lambda_)
            print 'err = ' + str(err)
            if(err < err_best):
                err_best = err
    print 'err_best= ' + str(err_best)
