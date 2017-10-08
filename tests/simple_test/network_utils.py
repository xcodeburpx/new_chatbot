import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, GaussianDropout, Activation, Input
from keras.activations import relu, elu, tanh, sigmoid, softmax, selu
from keras.initializers import Zeros, Ones, RandomNormal, TruncatedNormal, \
    VarianceScaling, Orthogonal, Identity

ACTIVATIONS = dict()
ACTIVATIONS["relu"] = relu
ACTIVATIONS['elu'] = elu
ACTIVATIONS['tanh'] =  tanh
ACTIVATIONS['sigmoid'] = sigmoid
ACTIVATIONS['selu'] = selu


INITIALIERS = dict()
INITIALIERS['zero'] = Zeros()
INITIALIERS['one'] = Ones()
INITIALIERS['random_normal'] = RandomNormal()
INITIALIERS['truncated_normal'] = TruncatedNormal()
INITIALIERS['variance_scaling'] = VarianceScaling()
INITIALIERS['orthogonal'] = Orthogonal()
INITIALIERS['identity'] = Identity()

def dense_model(input_shape, output_shape, hidden_list, activ_list,
                  output_activ,kernel_initalizer, bias_initalizer, optimizer, loss):

    model = Sequential()

    model.add(Dense(hidden_list[0], kernel_initializer=kernel_initalizer,
                    bias_initializer=bias_initalizer,input_shape=input_shape))
    model.add(Activation(ACTIVATIONS[activ_list[0]]))
    model.add(GaussianDropout(0.3))

    for h, a in zip(hidden_list[1:], activ_list[1:]):
        model.add(Dense(h, kernel_initializer=kernel_initalizer, bias_initializer=bias_initalizer))
        model.add(Activation(ACTIVATIONS[a]))
        model.add(GaussianDropout(0.3))

    model.add(Dense(output_shape[1], kernel_initializer=kernel_initalizer, bias_initializer=bias_initalizer))

    model.add(Activation(output_activ))
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return model

def test():
    train_data = np.random.randint(0, 2,[10000,100,20])
    input_shape = (train_data.shape[1], train_data.shape[2])

    model = dense_model(input_shape=input_shape,
                          output_shape=input_shape,
                          hidden_list=[100,50,20,50,100],
                          activ_list=['relu','elu','tanh','elu','relu'],
                          output_activ = "sigmoid",
                          kernel_initalizer='truncated_normal',
                          bias_initalizer='one',
                          optimizer='rmsprop',
                          loss='mse')

    print(model.summary(),"\n")
    time.sleep(2)
    model.fit(train_data, train_data,batch_size=64, epochs=50)

    test_data = np.random.randint(0, 2,[2000, 100, 20])
    #print(model.predict(test_data))


if __name__ == "__main__":
    test()
