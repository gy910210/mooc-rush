import math

def read_data(filename):
    x = []
    y = []
    infile = open(filename, 'r')
    for line in infile:
        strs = line.strip().split(' ')
        x.append((float(strs[0]), float(strs[1])))
        y.append(int(strs[2]))
    infile.close()
    return x, y

def decision_predict(x, i, theta, s):
    h = None
    if(x[i] - float(theta) >= 0):
        h = int(s)
    else:
        h = int(-s)
    return h

def decision_error(train_x, train_y, N, u, i, theta, s):
    err = 0.0
    for n in xrange(N):
        h = decision_predict(train_x[n], i, theta, s)
        if(h != int(train_y[n])):
            err += u[n]
    return float(err)

def decision_stump(train_x, train_y, u, N, d):
    i_opt = None
    s_opt = None
    theta_opt = None
    err_best = float("inf")

    for i in xrange(d):
        #every dimention
        tmp = []
        for n in xrange(N):
            tmp.append(train_x[n][i])
        sorted(tmp)

        theta_list = []
        theta_list.append(float("-inf"))
        for n in xrange(N - 1):
            theta_list.append(float(tmp[n]+tmp[n+1])/2.0)

        for n in xrange(N):
            err = None
            s = None
            err1 = decision_error(train_x, train_y, N, u, i, theta_list[n], 1)
            err2 = decision_error(train_x, train_y, N, u, i, theta_list[n], -1)
            if(err1 < err2):
                err = err1
                s = 1
            else:
                err = err2
                s = -1
            if(err < err_best):
                err_best = err
                i_opt = i
                s_opt = s
                theta_opt = theta_list[n]
    return i_opt, theta_opt, s_opt

def re_weight(train_x, train_y, N, u, i, theta, s, factor):
    u_new = []
    for n in xrange(N):
        h = decision_predict(train_x[n], i, theta, s)
        if(h == int(train_y[n])):
            u_new.append(u[n] / factor)
        else:
            u_new.append(u[n] * factor)
    return u_new

def adaboost_train(train_x, train_y, d, N, T):
    u = []
    for i in xrange(N):
        u.append(float(1.0/N))
    alpha = []
    model = []
    e_min = 10.0
    for t in xrange(T):
        #every t
        print('t = ' + str(t))
        i_opt, theta_opt, s_opt = decision_stump(train_x, train_y, u, N, d)
        model.append((i_opt, theta_opt, s_opt))

        err = decision_error(train_x, train_y, N, u, i_opt, theta_opt, s_opt)
        print('err = ' + str(err))
        weght_sum = 0.0
        for n in xrange(N):
            weght_sum += u[n]
        print 'weght_sum = ' + str(weght_sum)
        e = (float)(err / weght_sum)
        if(e < e_min):
            e_min = e
        print 'e = ' + str(e)
        factor = math.sqrt((1.0 - e) / e)

        u = re_weight(train_x, train_y, N, u, i_opt, theta_opt, s_opt, factor)
        alpha.append(math.log(factor))
    print 'e_min = ' + str(e_min)
    return alpha, model

def adaboost_predict(x, alpha, model, T):
    sum = 0.0
    for t in xrange(T):
        sum += alpha[t] * decision_predict(x, model[t][0], model[t][1], model[t][2])
    if(sum >= 0):
        return 1
    else:
        return -1

def adaboost_test(alpha, model, test_x, test_y, N, T):
    err = 0.0
    for n in xrange(N):
        h = adaboost_predict(test_x[n], alpha, model, T)
        if(h != test_y[n]):
            err += 1.0
    return float(err / float(N))
    
if __name__ == '__main__':
    train_x, train_y = read_data('data\\hw2_adaboost_train.dat')
    test_x, test_y = read_data('data\\hw2_adaboost_test.dat')
    
    d = 2
    N = len(train_x)
    T = 1
    alpha, model = adaboost_train(train_x, train_y, d, N, T)
    
    N = len(test_x)
    print ""
    test_err = adaboost_test(alpha, model, test_x, test_y, N, T)
    print "test error = " + str(test_err)