# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/train.png "train"
[image2]: ./examples/valid.png "valid"
[image3]: ./examples/test.png "test"
[image4]: ./web_images/00093.ppm "Traffic Sign 1"
[image5]: ./web_images/00099.ppm "Traffic Sign 2"
[image6]: ./web_images/00793.ppm "Traffic Sign 3"
[image7]: ./web_images/01762.ppm "Traffic Sign 4"
[image8]: ./web_images/01879.ppm "Traffic Sign 5"
[image9]: ./web_images/08788.ppm "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![train][image1]
![valid][image2]
![test][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Firstly I load data images from directory.

Then I turn the images to grayscale, this is because we'd better to reduce the data size and save processing time.

Then I normalize images to [0, 1] by applying `(pixel - 128)/ 128`. This is because it's better to have mean of 0 and equal variance. It won't change the data context, just make it easier to process data.

Lastly I shuffled the dataset to avoid any bias.

Since there is already a validation set. I don't need to split the training set. Thanks for that. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model `TrafficSign` consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU           		|           									|
| Max pooling   		| 2x2 stride, outputs 5x5x16					|
| Flatten       		| output 400   									|
| Fully connected		| input 400, output 120   						|
| RELU          		|           									|
| Dropout          		|           									|
| Fully connected		| input 120, output 84							|
| RELU          		|           									|
| Dropout          		|           									|
| Fully connected		| input 84, output 43							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I generate cross entropy using `tf.nn.softmax_cross_entropy_with_logits`. And calculate loss operation using `tf.reduce_mean`.
In TensorFlow, there is an `tf.train.AdamOptimizer` to help us optimize model.

To train the model, I used parameters as following:
rate = 0.0012
EPOCHS = 50
BATCH_SIZE = 128


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of  0.967
* test set accuracy of 0.945

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    
    I used the code from LeNet lab in the lecture. Because it's almost the same use cases. Then I made some updates to it to fit in our problem here.
     
* What were some problems with the initial architecture?
       
    1. Number of classes are different.
    
    2. Iutput dimentions are different. 
    
    3. Acurracy is not good enough.
    
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

    I did a lot of trying of training the updated LeNet code but it didn't work well. The accuracy is at most 92%. 
    
    So I did some research in the course resources and online. Finally I decided to add `tf.nn.dropout()`. 
    This forces the network learn using redundant representation for everything by applying a drop probability of 50%. It can make the network more robust.
    
* Which parameters were tuned? How were they adjusted and why?
    
    I tried a lot of tuning the learning rate and epochs.
    
    Firstly, the learning rate was 0.01 and epoch 40. But it's not good enough, after a lot of back and forth, finally I made them 0.0012 and 50 epoch.
    
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![14][image4] ![33][image5] ![3][image6] 
![17][image7] ![25][image8] ![7][image9]

The 1st image might be difficult to classify because there is another black line there;
The 2nd image might be difficult to classify because it's almost pure blue;
The 3rd image might be difficult to classify because it's dark;
The 4th image might be difficult to classify because it's distored;
The 5th image might be difficult to classify because it's blured;
The 6th image might be difficult to classify because I want to test a speed limit sign;

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Turn right ahead    	| Turn right ahead 								|
| Speed limit (60km/h)	| Speed limit (60km/h)							|
| No entry	      		| No entry					 	    			|
| Road work 			| Road work       				    			|
| Speed limit (100km/h)	| Speed limit (100km/h)     		    		|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

I used `sess.run(tf.nn.top_k(tf.constant(a), k=5))` method to generate the values and indices (class ids) of the top 5 predictions. 

The raw output is like below, which is showing in the ipynb file and HTML output as well:

```
values=array([[  1.00000000e+00,   6.62047639e-11,   2.55759519e-12,
          3.28098524e-13,   2.39975628e-13],
       [  1.00000000e+00,   2.38301930e-11,   4.31474257e-12,
          6.93528808e-15,   5.04318156e-17],
       [  9.99809444e-01,   1.90580075e-04,   1.56659041e-09,
          8.68713435e-10,   2.04789213e-10],
       [  9.99847531e-01,   1.52236957e-04,   6.42974598e-08,
          5.55156952e-08,   4.10474463e-08],
       [  1.00000000e+00,   3.99670511e-17,   2.51116003e-17,
          1.12691344e-18,   2.36870794e-19],
       [  7.12511241e-01,   2.78924465e-01,   7.81362597e-03,
          6.85630075e-04,   5.55059814e-05]], dtype=float32), indices=array([[14, 33, 12,  4, 13],
       [33, 14, 35, 13, 39],
       [ 3,  5, 42,  2,  6],
       [17, 14, 33,  4, 39],
       [25, 20, 22, 29, 24],
       [ 7, 40,  1,  8, 12]], dtype=int32))
       
To make is more readable, I make two talbes here(just for demo, not in code):
       
Top 5 softmax probabilities for each image:

| 1              | 2               |    3	   	     |  4              | 5               |
|:--------------:|:---------------:|:---------------:|:---------------:|:---------------:| 
|  1.00000000e+00|   4.52442350e-09|   3.62675689e-09|   7.91167132e-10|   6.60578481e-10|
|  9.99998689e-01|   1.32900925e-06|   5.41109033e-11|   4.88772694e-12|   2.20905455e-12|
|  9.99850750e-01|   1.48977313e-04|   1.99902985e-07|   4.51238407e-08|   1.30417535e-08|
|  9.99999642e-01|   4.06699485e-07|   2.73375367e-08|   4.08073619e-09|   1.67154290e-09|
|  1.00000000e+00|   2.19890100e-12|   5.06858993e-14|   9.65517610e-15|   8.74828333e-15|
|  9.04017389e-01|   8.57333392e-02|   7.15107238e-03|   1.90547458e-03|   5.75917889e-04|

Top 5 results(traffic sign lables):

| 1              | 2               |    3	   	     |  4              | 5               |
|:--------------:|:---------------:|:---------------:|:---------------:|:---------------:| 
|14              | 34              | 15              | 17              | 39              |
|33              | 39              | 14              | 35              | 13              |
|3               |  5              | 23              | 10              | 16              |
|17              | 14              |  0              | 34              | 36              |
|25              | 20              | 29              | 22              | 30              |
| 7              | 40              | 12              |  1              | 16              |

They are predicted pretty good. Accuracy is 100% and certenty is high.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


