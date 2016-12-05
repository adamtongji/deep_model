import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(x,y,weight, learning_rate, samples, maxiterations ,batch=1):
    loss_cache=[]
    xtrains = x.transpose()
    for i in range(maxiterations):
        pred = np.dot(x, weight)
        loss = pred - y
        # gradient for weight = x.T * loss matrix
        gradient = np.dot(xtrains, loss) /samples
        weight -= learning_rate* gradient
        loss_cache.append(np.mean(loss))
        if i%100==0:
            print np.mean(loss), "iterate {0} times".format(i)
    return weight, loss_cache



def get_data(dataset):
    m,n =np.shape(dataset)
    traindata = np.zeros((m,n))
    traindata[:,:-1]= dataset[:,:-1]
    trainlabel = dataset[:,-1]
    return traindata, trainlabel




def predict(x, weight):
    m, n =np.shape(x)



dat = np.random.randn(100, 32)
label = np.random.randint(10, size=100)

m,n =np.shape(dat)
l_rate = 0.01
# weights = np.random.random(n)
weights= np.ones(n)
iters= 3099

final_weight, loss_total=gradient_descent(dat, label, weights, l_rate,m, iters)

plt.subplot(211)
plt.ylim((-2,2))

plt.plot(loss_total)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()