# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/X_train_hist_before.png "Visualization"
[image2]: ./examples/Traffic-Sign-grayscale-after.png "Grayscaling After"
[image3]: ./examples/Augmented-Sign.png "Augmented Sign"
[image4]: ./new_pictures/5.png "Traffic Sign 1"
[image5]: ./new_pictures/1.png "Traffic Sign 2"
[image6]: ./new_pictures/3.png "Traffic Sign 3"
[image7]: ./new_pictures/4.png "Traffic Sign 4"
[image8]: ./new_pictures/2.png "Traffic Sign 5"
[image9]: ./examples/Traffic-Sign-grayscale-before.png "Grayscaling Before"

---

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Rxmeez/SDCND-1-TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the default python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed over the 43 classes.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because computation naturally would be faster as this is comparing a single dimension with 3 color channels. Secondly color information doesn't help identify important edges and other features. So by turning the image to grayscale also reduces the complexity as we only deal with 1 dimension, where the grayscale would only be effected based on luminance.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image9] ![alt text][image2]

As a last step, I normalized the image data because having the value range from 0.1 to 0.9 meant the all values would be in a similar range even though for an image it would always be in a fixed range 0 - 255. Where normalization would have helped is in the computation where its constantly being multiplied to calculate gradient error vectors.

I decided to generate additional data because a general trend is the more data a model will have the better it would perform and looking into the distribution I wanted to add data to the classes which had relatively low data.

To add more data to the the data set, I used the following techniques because having a look at the training data of 10 random images, I could see there is a variation in the brightness of images, it had different positioning, rotation, blur/noise when I compared a single class.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following 11915 where the new training dataset is now 46714.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 GRAY image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| activation function												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x6      									|
| RELU          | activation function                       |
| Max pooling         | 2x2 stride, outputs 5x5x16          |
| Flatten             | outputs 400                         |
| Fully connected		| outputs 120        									|
| RELU              | activation function                   |
| Dropout           | regularization                  |
| Fully connected   | outputs 84        |
| RELU              | activation function                  |
| Dropout           | regularization                   |
| Fully connected				| outputs 43        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimzer and after running different cases I found the optimal settings to give me the following hyperparameters.
* Number of EPOCH = 25
* Batch Size = 200
* Learning rate = 0.0015
* Dropout = 0.6

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.947
* validation set accuracy of 0.953
* test set accuracy of 0.940

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

A standard architecture was chosen, LeNet with a slight modification which are the input wasnt a RGB but a GRAY image so the dimension was 1, with filters 5x5, a dropout was added in the fully connected this regularization will help overfitting the data.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The images where standard random traffic signs pulled from google, the difficultly that the classifier might face is to distinguish between the speed signs when it wasnt trained on a balanced distributed graph, so some signs will skew and to test this the 20km/h sign is placed.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (20km/h)      		| Speed limit (50km/h)   									|
| No Entry     			| No Entry 										|
| Keep left					| Keep left											|
| Roundabout mandatory	      		| Roundabout mandatory					 				|
| Double curve			| Road narrows on the right      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This don't compares favorably to the accuracy on the test set of 94%, one of the reasons I believe this could be is that the model trained on certain images like speed limit (20km/h) were of low data compared to others. One solution could be if I augmented data for all classes to be equal, this will prevent certain signs being skewed towards, in this case speed limit (50km/h).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is wrong that this is a Speed limit (50km/h) (probability of 0.46), and the image contain a Speed limit (20km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        	|
|:---------------------:|:-------------------------:|
| .46         			    | Speed limit (50km/h)   							|
| .21     				      | Speed limit (20km/h) 										|
| .14					          | Double curve											|
| .10	      			      | Speed limit (60km/h)					 			|
| .09				            | Slippery road      				|


For the second image, the model is correct that this is a No Entry (probability of 0.54), and the image contain a No Entry. The top five soft max probabilities were

| Probability         	|     Prediction	        	|
|:---------------------:|:-------------------------:|
| .54         			    | No Entry   							|
| .20     				      | Stop 										|
| .13					          | Vehicles over 3.5 metric tons prohibited											|
| .08	      			      | Speed limit (20km/h)					 			|
| .05				            | Turn left ahead      				|

For the third image, the model is correct that this is a Keep left (probability of 0.46), and the image contain a Keep left. The top five soft max probabilities were

| Probability         	|     Prediction	        	|
|:---------------------:|:-------------------------:|
| .46         			    | Keep left   							|
| .14     				      | End of no passing			|
| .14					          | End of all speed and passing limits|
| .13	      			      | End of no passing by vehicles over 3.5 metric ...|
| .13				            | End of speed limit (80km/h)|

For the fourth image, the model is correct that this is a Roundabout mandatory (probability of 0.42), and the image contain a Roundabout mandatory. The top five soft max probabilities were

| Probability         	|     Prediction	        	|
|:---------------------:|:-------------------------:|
| .42         			    | Roundabout mandatory|
| .23     				      | Speed limit (100km/h)	|
| .18					          | Priority road	|
| .09	      			      | Go straight or left 			|
| .08				            | End of no passing by vehicles over 3.5 metric ...	|

For the fifth image, the model is wrong that this is a Road narrows on the right (probability of 0.28), and the image contain a Double curve. The top five soft max probabilities were

| Probability         	|     Prediction	        	|
|:---------------------:|:-------------------------:|
| .28         			    | Road narrows on the right|
| .20     				      | Beware of ice/snow|
| .19					          | Pedestrians								|
| .17	      			      | Bicycles crossing			 			|
| .16				            |  Road work   				|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
