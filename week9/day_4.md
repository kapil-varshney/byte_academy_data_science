## 2.0 LSTM Networks

Long Short Term Memory Networks are a special kind of recurrent neural network capable of learning long-term dependencies. LSTMs are explicitly designed to avoid the long-term dependency problem, so remembering information for long periods of time is usually their default behavior.

LSTMs also have the same RNN chain-like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very particular way. You can see that [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png). Compared to the typical recurrent neural network [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png), you can see there's a much more complex process happening. 


### 2.1 First Step

The first step in a LSTM is to decide what information is going to be thrown away from the cell state. This decision is made by a <b>sigmoid layer</b> called the “forget gate layer.” It outputs a number between 00 and 11 for each number in the cell state, where a 1 represents disposal and a 0 means storage.

In the context of natural language processing, the cell state would include the gender of the present subject, so that the correct pronouns can be used in the future.

### 2.2 Second Step

This next step is deciding what new information is going to be stored in the cell state. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Secondly, a tanh layer creates a vector of new values that <i>could</i> be added to the cell state. 

In our NLP example, we would want to add the gender of the new subject to the cell state, to replace the old one we’re forgetting.

### 2.3 Third Step

So now we update the old cell state. The previous steps already decided what to do, but we actually do it in this step.

We multiply the old state by new state function, causing it to forget what we've learned earlier. Then we add the updates!

In the case of the language model, this is where we’d actually drop the information about the old subject’s gender and add the new information, which we decided in the previous steps.

### 2.4 Final Step

Finally, we need to decide what we’re going to output. This output will be based on our cell state, but only once we've filtered it. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through tanhtanh (to push the values to be between −1 and 1) and multiply it by the output of the sigmoid gate, so that we only output the parts we need to.

For our NLP example, since it just saw a subject, it might want to output information relevant to a verb, in case that’s what is coming next. For example, it might output whether the subject is singular or plural, so that we know what form a verb should be conjugated into if that’s what follows next.

## 6.0 Convolution Neural Networks

Convolutional neural networks are a type of feed-forward networks, which perform very well on visual recognition tasks. 

A typical feature of CNN’s is that they nearly always have images as inputs, this allows for more efficient implementation and a reduction in the number of required parameters. 

There are two basic principles that define convolution neural networks: filters and characteristic maps. 

The main purpose of a convolutional layer is to detect characteristics or visual features in the images, such as edges, lines, colors, etc. This is done by a hidden layer connected to the input layer.

This step, called the convolution step, can be shown [here](https://ujwlkarn.files.wordpress.com/2016/07/convolution_schematic.gif?w=536&h=392). In CNNs, the 3×3 matrix is called the 'filter' or 'kernel' and the matrix formed by sliding the filter over the image and computing the dot product is called the ‘Convolved Feature’ or ‘Activation Map’ or the ‘Feature Map‘. It is important to note that filters act as feature detectors from the original input image.

Now, let’s begin our MNIST digit recognition example. So naturally, first we import the needed modules: 

``` python
import tensorflow as tf
import input_data
```

And of course, the actual dataset: 
``` python 
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

As before, we'll define the placeholders using TensorFlow as we did in the exercise in section 3.2.
``` python
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
```

We can reconstruct the original shape of the images of the input data. We can do this as follows:

``` python
x_image = tf.reshape(x, [-1,28,28,1])
```

In order to simplify the code, I define the following two functions related to the weight matrix and bias:


``` python
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
```

Similar to above, we define these two generic functions to be able to write a cleaner code that involves convolutions and max-pooling.

Spatial Pooling reduces the dimensionality of each feature map but retains the most important information. In case of Max Pooling, a type of spatial pooling, we define a spatial neighborhood and take the largest element from the rectified feature map within that window. 

``` python
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```

Now it is time to implement the first convolutional layer followed by a pooling layer. 
``` python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```

As a final step, we apply max-pooling to the output:
``` python
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```

When constructing a deep neural network, we can stack several layers on top of each other. 

``` python
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
```

Now, we want to flatten the tensor into a vector.
``` python
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
```

Now we are ready to train the model that we have just defined by adjusting all the weights in the convolution, and fully connected layers to obtain the predictions of the images. 


``` python
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

sess.run(tf.initialize_all_variables())

for i in range(200):
batch = mnist.train.next_batch(50)
if i%10 == 0:
train_accuracy = sess.run( accuracy, feed_dict={ x:batch[0], y_: batch[1], keep_prob: 1.0})
print("step %d, training accuracy %g"%(i, train_accuracy))
sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"% sess.run(accuracy, feed_dict={ 
x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

## 4.0 Convolution Neural Networks

Convolutional Neural Network (CNNs) are typically thought of in terms of Computer Vision, but they have also had some important breakthroughs in the field of Natural Language Processing. In this section, we'll overview how CNNs can be applied to Natural Language Processing instead.

### 4.1 Input

Instead of image pixels, the input for an NLP model will be sentences or documents represented as a matrix. Each row of the matrix corresponds to one token. These vectors will usually be word embeddings, discussed in section 1 of this workshop, like word2vec or GloVe, but they could also be one-hot vectors that index the word into a vocabulary. For a 10 word sentence using a 100-dimensional embedding we would have a 10×100 matrix as our input. That's what replaces our typical image input.

### 4.2 Filters

Previously, our filters would slide over local patches of an image, but in NLP we have these filters slide over full rows of the matrix (remember that each row is a word/token). This means that the “width” of our filters will usually be the same as the width of the input matrix. The height may vary, but sliding windows over 2-5 words at a time is typical. 

Putting all of this together, here's what an NLP Convolution Neural Network would look [like](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/11/Screen-Shot-2015-11-06-at-12.05.40-PM.png). 

### 4.3 Tensorflow Implementation

In this portion of the tutorial, we'll be implementing a CNN for text classification, using Tensorflow. 

First, begin by importing the needed modules:

``` python
import tensorflow as tf
import numpy as np
```

In this implementation, we'll allow hyperparameter configurations to be customizable so we'll create a TextCNN class, generating the model graph in the init function.


``` python
class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):
```

Note the needed arguments to instantiate the class:

- sequence_length: The length of our sentences
- num_classes: Number of classes in the output layer, two in our case.
- vocab_size: The size of our vocabulary. 
- embedding_size: The dimensionality of our embeddings.
- filter_sizes: The number of words we want our convolutional filters to cover.
- num_filters: The number of filters per filter size.


We then officially start by defining the input data that we pass to our network:

``` python
self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
```

The tf.placeholder creates a placeholder variable that we feed to the network when we execute it at train or test time. The second argument is the shape of the input tensor. None means that the length of that dimension could be anything. In our case, the first dimension is the batch size, and using None allows the network to handle arbitrarily sized batches.

#### Embedding Layer

The first layer we define is the embedding layer, which maps vocabulary word indices into low-dimensional vector representations. 

``` python
with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
        name="W")
    self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
    self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
```

- tf.device("/cpu:0"): This forces an operation to be executed on the CPU. By default TensorFlow will try to put the operation on the GPU if one is available, but the embedding implementation doesn’t currently have GPU support and throws an error if placed on the GPU.
- tf.name_scope: This creates a new Name Scope with the name “embedding”. The scope adds all operations into a top-level node called “embedding” so that you get a nice hierarchy when visualizing your network in TensorBoard.
- W: This is our embedding matrix that we learn during training. We initialize it using a random uniform distribution. 
- tf.nn.embedding_lookup: This creates the actual embedding operation. The result of the embedding operation is a 3-dimensional tensor of shape [None, sequence_length, embedding_size].


Now we’re ready to build our convolutional layers followed by max-pooling. Because each convolution produces tensors of different shapes we need to iterate through them, create a layer for each of them, and then merge the results into one big feature vector.

``` python
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            self.embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Max-pooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs.append(pooled)
```
Then, we combine all the pooled features:

``` python
num_filters_total = num_filters * len(filter_sizes)
self.h_pool = tf.concat(3, pooled_outputs)
self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
```

- W: This is our filter matrix.
- h: This is the result of applying the nonlinearity to the convolution output. Each filter slides over the whole embedding, but varies in how many words it covers. 
- "VALID": This padding means that we slide the filter over our sentence without padding the edges, performing a narrow convolution that gives us an output of shape [1, sequence_length - filter_size + 1, 1, 1]. 
- Performing max-pooling over the output of a specific filter size leaves us with a tensor of shape [batch_size, 1, 1, num_filters]. This is essentially a feature vector, where the last dimension corresponds to our features. Once we have all the pooled output tensors from each filter size we combine them into one long feature vector of shape [batch_size, num_filters_total]. 
- Using -1 in tf.reshape tells TensorFlow to flatten the dimension when possible.

### Dropout Layer

Dropout is the perhaps most popular method to regularize convolutional neural networks. A dropout layer stochastically disables a fraction of its neurons, which prevents neurons from co-adapting and forces them to learn individually useful features. The fraction of neurons we keep enabled is defined by the dropout_keep_prob input to our network. 

``` python
with tf.name_scope("dropout"):
    self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
```

Using the feature vector from max-pooling (with dropout applied) we can generate predictions by doing a matrix multiplication and picking the class with the highest score. We could also apply a softmax function to convert raw scores into normalized probabilities, but that wouldn’t change our final predictions.

``` python
with tf.name_scope("output"):
    W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
    self.predictions = tf.argmax(self.scores, 1, name="predictions")
```

- tf.nn.xw_plus_b: This is a convenience wrapper to perform the Wx + b matrix multiplication.


We can now define the loss function., which is a measurement of the error our network makes. Remember that our goal is to minimize this function. The standard loss function for categorization problems is the cross-entropy loss, which we implement here:

``` python
with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
    self.loss = tf.reduce_mean(losses)
``` 

-tf.nn.softmax_cross_entropy_with_logits: This is the convenience function that calculates the cross-entropy loss for each class, given our scores and the correct input labels. We then take the mean of the losses. We could also use the sum, but that makes it harder to compare the loss across different batch sizes and train/dev data.

We also define an expression for the accuracy, which is a useful quantity to keep track of during training and testing.

``` python
with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
```

And so now we’re done with our network definition!


#### Training

First, we begin by instantiating our model. By doing so, all of our variables and operations will be placed into our default graphs and sessions. 

``` python 
cnn = TextCNN(
    sequence_length=x_train.shape[1],
    num_classes=2,
    vocab_size=len(vocabulary),
    embedding_size=FLAGS.embedding_dim,
    filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
    num_filters=FLAGS.num_filters)
```

Next, we define how to optimize our network’s loss function. TensorFlow has several built-in optimizers. We’re using the Adam optimizer.

``` python
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-4)
grads_and_vars = optimizer.compute_gradients(cnn.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
```

- train_op: This is a newly created operation that we can run to perform a gradient update on our parameters. Each execution is a training step. TensorFlow automatically figures out which variables are “trainable” and calculates their gradients. 
- global_step: By passing this to the optimizer we allow TensorFlow handle the counting of training steps for us. The global step will be automatically incremented by one every time you execute train_op.


Now, before we can train our model we also need to initialize the variables in our graph.

``` python
sess.run(tf.initialize_all_variables())
```

Let’s now define a function for a single training step, evaluating the model on a batch of data and updating the model parameters.

``` python
def train_step(x_batch, y_batch):

    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    train_summary_writer.add_summary(summaries, step)
```

- feed_dict: This contains the data for the placeholder nodes we pass to our network. You must feed values for all placeholder nodes, or TensorFlow will throw an error.
-train_op: This actually returns nothing, it just updates the parameters of our network. 


Finally, we’re ready to write our training loop. First we initialize the batches:

``` python
batches = data_helpers.batch_iter(
    zip(x_train, y_train), FLAGS.batch_size, FLAGS.num_epochs)
```

Then we iterate over batches of our data, call the train_step function for each batch, and occasionally evaluate and checkpoint our model:

``` python
for batch in batches:
    x_batch, y_batch = zip(*batch)
    train_step(x_batch, y_batch)
    current_step = tf.train.global_step(sess, global_step)
    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        dev_step(x_dev, y_dev, writer=dev_summary_writer)
        print("")
    if current_step % FLAGS.checkpoint_every == 0:
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))
```



