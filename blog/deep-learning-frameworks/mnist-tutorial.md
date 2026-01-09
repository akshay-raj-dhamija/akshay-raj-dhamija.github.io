# MNIST Caffe2 Tutorial

**2017-05-27 00:00** | *Caffe2*

[← Back to Blog](../index.html)

---

Mnist using caffe2 with Specific GPU use 

Importing general packages

Importing Caffe2 Packages

##### Beaming Up Protocol
The requiered protocol files may be downloaded from https://www.kaggle.com/c/digit-recognizer/data

##### Preprocessing Protocol
Let's create a 80/20 split of the provided training data to generate a validation set.<br>
The new dataset structure can be seen in the output.

**Output:**
```
(33600, 785) (8400, 785)

```

##### Defining LeNet Architecture
Since we wish to run the experiments on the GPU lets specify the device option.

Now, lets define the LeNet architecture<br>
Note: in caffe2 currently each layer requiers inputs dimension size too

Calculating Accuracy

##### Defining the Training Operators
Here we calculate the cross entropy loss and call the accuracy function to calculate the accuracy on the training set.<br>
Most importantly we add the loss as a gradient in order to enable back propogation.<br>
We also initialize the SGD solver along with the learning rate policy we follow.<br>
Since, the learning rate policy is `step` and `stepsize` is 1 at every new iteration the learning rate would be<br>

$$Base\ Learning\ Rate*\gamma^{iteration\ no - 1}$$

Let's define the batch size

Lets Reset the Workspace

##### Defining Training Model
Let's create a model that we will use for training.

Define the GPU ID to run experiment on

To run on single GPU of choice

Adding network and training operators to the training model

Initializing the network and loading it into the workspace

##### Saving snapshots
The following function saves layer weights for later use. <br>
The weights are saved in a python dictionary with the keys as the blob name and value as the weights.
Note, better ways exists at https://github.com/caffe2/caffe2/blob/master/caffe2/python/predictor/mobile_exporter.py

##### Creating Validation Model
Lets create and define the validation model similar to the training model mentioned above.<br>
Note, here we don't initialize the weights but use whatever are the existing values for the weights. We also don't add the training operators therefore there won't be any loss for the validation set.

Let's evaluate the performance of a given model on the entire validation set.<br>
the function returns the averaged loss and accuracy

Let's define the interval at which we intend to take snapshots

##### Running Training
We train the model on the training set and evaluate its performance on validation set after each iteration

Lets evaluate the accuracy on the training and validation set for each iteration

Lets evaluate the changes in loss values for each iteration of the training set

Note: Since for internal processing caffe2 negates the learning rate it currently returns negative of the actual learning rate, therefore we will inverse the signs before visualizing.

##### Creating Testing Model
We create the test model, to predict results for the test set.<br>
Note, that in this case we dont calculate the accuracy and in the input blobs we don't provide the labels

Lets find the iteration that performs best on the validation set and for which we have a snapshot

Lets feed the best weight combination found above into the workspace

Let's predict the output for the test set

Beaming Down Results

Upload the csv file created above at https://www.kaggle.com/c/digit-recognizer/submit to evaluate the performance on the test set.<br>
You should obtain an accuracy greater than 95% on the test set.<br>

This completes the MNIST experiment


---

[← Back to Blog](../index.html)
