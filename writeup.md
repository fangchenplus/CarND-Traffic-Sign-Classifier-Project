    # **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # "Image References"

[image1a]: ./result_pictures/samples_training_set.png "Visualization"
[image1b]: ./result_pictures/distribution_training_set.png "Visualization"
[image2]: ./LeNet.jpg "LeNet-5"
[image3]: ./result_pictures/accuracy.png "Validation set accuracy"
[image4]: ./result_pictures/my_found_signs.png "My Found Signs"
[image5a]: ./result_pictures/feature_map_1.png "Feature map 1"
[image5b]: ./result_pictures/feature_map_2.png "Feature map 2"
[image5c]: ./result_pictures/feature_map_3.png "Feature map 3"
[image5d]: ./result_pictures/feature_map_4.png "Feature map 4"
[image5e]: ./result_pictures/feature_map_5.png "Feature map 5"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---

### Writeup / README

Here is a link to my [project code](https://github.com/fangchenplus/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) and [html file](https://github.com/fangchenplus/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy function `.shape` and `len()` to calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Some random traffic signs from the training set are drawn below. The bar chart shows the distribution of the 43 classes in the training set. It is clearly the training data is unevenly distributed across the classes.  It is better to use data augmentation to balance the data.

![alt text][image1a]

![alt text][image1b]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

It is important to preprocess the image data before training the neutral network. In this project, I considered normalization and grayscaling. Since the training data set given in this project is heavily biased to some of the classes, it will also be helpful to use image augmentation to generate new images for the classes with fewer training images. The technique can be image zoom in/out, flip, horizontal and vertical offset etc. Due to time limitation, only normalization and grayscale were attempted in this project, and still achieved an accuracy > 93%. 

In fact, grayscaling helps to reduce input image data size by three times, but at the same time it loses the color information in the original images. After several attempts, I found grayscale does not bring much benefit in this particular project. So in the final version, only normalization is adopted, by using `x = (x.astype('float32')-128)/128` to make all the input data within the range of [-1, 1]. 

After normalization, the image is complete dark by calling `plt.imshow()` since the function only accept [0,1] for floats and [0,255] for integers.

Last but not least, all the input data is shuffled before feeding into the CNN by running `X_train, y_train = shuffle(X_train, y_train)`

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The LeNet showing below (cited from the [original paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)) is the structure used in this project. The LeNet-5 consists of two convolution layers and three fully connected layers.

![alt text][image2]

My final model consisted of the following layers:

|          Layer          |                 Description                 |
| :---------------------: | :-----------------------------------------: |
|          Input          |              32x32x3 RGB image              |
|   Convolution 5x5x3x6   | 1x1 stride, valid padding, outputs 28x28x6  |
|          RELU           |               outputs 28x28x6               |
|     Max pooling 2x2     |        2x2 stride,  outputs 14x14x6         |
|  Convolution 5x5x6x16   | 1x1 stride, valid padding, outputs 10x10x16 |
|          RELU           |              outputs 10x10x16               |
|     Max pooling 2x2     |         2x2 stride,  outputs 5x5x16         |
|         Flatten         |                 outputs 400                 |
| Fully connected 400x120 |                 outputs 120                 |
|          RELU           |                 outputs 120                 |
| Fully connected 120x84  |                 outputs 84                  |
|          RELU           |                 outputs 84                  |
|  Fully connected 84x43  |                 outputs 43                  |
|          RELU           |                 outputs 43                  |
|         Softmax         |              final outputs 43               |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following training parameters and pipeline.

| Parameter | Value  |
| :-------: | :-------: |
|      Optimizer     |    AdamOptimizer           |
| Batch size | 128 |
| Number of epochs | 50 |
| Learning rate | 0.001 |

```python
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

* What architecture was chosen? **LeNet-5**
* Why did you believe it would be relevant to the traffic sign application? The model is simple and has been used to recognize numbers. Considering the similarity between numbers and traffic signs, LeNet is a good candidate for this project.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

After 50 epochs, my model reached a **94%** accuracy for validation set and **93.3%** accuracy for test set. The trend of validation accuracy versus epochs is drawn below. The flat accuracy after around 40 epochs indicates a good settling with no underfitting or overfitting. Thus no dropout layer is added to the model.

![alt text][image3]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web (size of 32x32x3). 

![alt text][image4]

Most of the signs are visually distinguishable. The first and second signs in the second row both have a triangle shape, thus could be misclassified. The speed signs are also sometime misclassified, due to their similar shape and color. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|                 Image                  |               Prediction               |
| :------------------------------------: | :------------------------------------: |
|            Turn left ahead             |            Turn left ahead             |
|                  Stop                  |                  Stop                  |
|          Speed limit (60km/h)          |          Speed limit (60km/h)          |
|               Keep right               |               Keep right               |
|                 Yield                  |                 Yield                  |
|               Road work                |               Road work                |
| Right-of-way at the  next intersection | Right-of-way at the  next intersection |
|          Speed limit (30km/h)          |          Speed limit (30km/h)          |
|             Priority road              |             Priority road              |
|            General caution             |            General caution             |

The model was able to correctly classify all 10 traffic signs, which gave an accuracy of 100%. In my test, sometimes it missed one of the classification, which dropped the accuracy to 90%. Nevertheless, this demonstrates the accuracy of the trained traffic sign classifier.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the Ipython notebook section named **Output Top 5 Softmax Probabilities For Each Image Found on the Web**.

It can be observed that in all ten cases, the model has approximately 1.0 predicted probability for the correct answer, which proves the fidelity of the model. 

| Probability |               Prediction               |
| :---------: | :------------------------------------: |
|     1.0     |            Turn left ahead             |
|     1.0     |                  Stop                  |
|   0.9999    |          Speed limit (60km/h)          |
|     1.0     |               Keep right               |
|     1.0     |                 Yield                  |
|     1.0     |               Road work                |
|     1.0     | Right-of-way at the  next intersection |
|  0.9999998  |          Speed limit (30km/h)          |
|     1.0     |             Priority road              |
|     1.0     |            General caution             |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The feature maps of different layers of the neural network are drawn below to observe what characteristics did the model capture. The first traffic sign (Turn left ahead) is used an example input.

The feature map (weight) of the first convolution layer is drawn below. As expected, the layer recognized the round shape and the left turn arrow.

![alt text][image5a]

The feature map of the second convolution layer is below. Theoretically, it is expect to capture high-level features of the sign. But visually, it is hard to recognize.

![alt text][image5b]

The feature maps of the three fully connected layers are also drawn below sequentially. It can be observed that the bar chart of the last layer has high weighting factors for the outputs with high probability. This reflects the classification result of the CNN of the first sign.

![alt text][image5c]

![alt text][image5d]

![alt text][image5e]

