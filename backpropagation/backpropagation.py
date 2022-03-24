from random import uniform
import numpy as np

X=np.array(([2, 9], [1, 5], [3, 6]))
X=X/np.amax(X,axis=0)

Y = np.array(([92], [86], [89]))
Y = Y/100

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
    return(x*(1-x))


epoch=7000
learning_rate=0.1
input_layer_neuron=2
hidden_layer_neuron=3
ouptput_layer_neuron=1

wh=np.random.uniform(size=(input_layer_neuron,hidden_layer_neuron))
bh=np.random.uniform(size=(1,hidden_layer_neuron))
wout=np.random.uniform(size=(hidden_layer_neuron,ouptput_layer_neuron))
bout=np.random.uniform(size=(1,ouptput_layer_neuron))

for i in range(epoch):
    #forward propagation
    xinp=np.dot(X,wh)+bh
    hidden_act=sigmoid(xinp)

    outinp=np.dot(hidden_act,wout)+bout
    output=sigmoid(outinp)

    #backward propagation
    EO=Y-output
    outgrad=derivative_sigmoid(output)
    d_output=EO*outgrad

    EH=np.dot(d_output,wout.T)
    hidgrad=derivative_sigmoid(hidden_act)
    d_hidden=EH*hidgrad

    wout+=np.dot(hidden_act,d_output)*learning_rate
    wh+=np.dot(X.T,d_hidden)*learning_rate

print('Input: ',X)
print('Actual output: ',Y)
print('Predicted output: ', output)