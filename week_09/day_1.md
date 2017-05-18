## 4.0 Feedforward Neural Networks 

Feedforward Neural Networks are the simplest form of Artificial Neural Networks. These networks have the three types of layers we just discussed: Input layer, hidden layer and output layer. There are no backwards or inter-layer connections. Furthermore, the nodes in the layer are fully connected to the nodes in the next layer. 

First, we import the needed modules.

``` python
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os
```

This function resizes the image to a fixed size and then flattens it into a list of features (raw pixel intensities).

``` python
def image_to_feature_vector(image, size=(32, 32)):
    return (cv2.resize(image, size).flatten())
```

Here, we construct the argument parse and parse the arguments.

``` python
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
args = vars(ap.parse_args())
``` 

So here we grab the list of images that we'll be describing and initialize the data matrix and labels list.
``` python
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

data = []
labels = []
```

Iterating over the input images, we load the image and extract the class label. Secondly, we construct a feature vector of raw pixel intensities and then update the matrix and list. 
``` python
# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    features = image_to_feature_vector(image)
    data.append(features)
    labels.append(label)
 
    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))
```


``` python
le = LabelEncoder()
labels = le.fit_transform(labels)

data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 2)
 
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data, labels, test_size=0.25, random_state=42)
```


Here, we're just defining the architecture of the network

``` python
model = Sequential()
model.add(Dense(768, input_dim=3072, init="uniform",
    activation="relu"))
model.add(Dense(384, init="uniform", activation="relu"))
model.add(Dense(2))
model.add(Activation("softmax"))
```

Now, we begin training the model: 
``` python 
print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd,
    metrics=["accuracy"])
model.fit(trainData, trainLabels, nb_epoch=50, batch_size=128)
```

``` python
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
    batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
    accuracy * 100))
```

## 5.0 Recurrent Neural Networks

As we said in section 1, a Recurrent Neural Network is a class of ANNs where connections between units form a directed cycle, as shown [here](https://www.talaikis.com/wp-content/uploads/2016/03/rl.png).

Because recurrent neural networks form directed cycles, information is able to persist, meaning it can use its reasoning from previous events. A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. 


``` python
import copy, numpy as np
np.random.seed(0)
```

This helper function just computes the sigmoid nonlinearity.
``` python
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
```


Here, this function converts the output of the sigmoid function to its derivative.
``` python 
def sigmoid_output_to_derivative(output):
    return output*(1-output)
``` 

Obviously we need a dataset to train on, so this is where we generate that: 
``` python
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]
```

Here, we define our dimension variables!

``` python
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1
```

And, as always, we initialize neural network weights:
``` python
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)
```

Finally, we begin our training. 
``` python
for j in range(10000):
    
    a_int = np.random.randint(largest_number/2) 
    a = int2binary[a_int] 

    b_int = np.random.randint(largest_number/2) 
    b = int2binary[b_int] 

    c_int = a_int + b_int
    c = int2binary[c_int]
    
    d = np.zeros_like(c)

    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    for position in range(binary_dim):
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    
    future_layer_1_delta = np.zeros(hidden_dim)
    
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # update weights for back-propagation
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(a_int) + " + " + str(b_int) + " = " + str(out)
        print "------------"

```
