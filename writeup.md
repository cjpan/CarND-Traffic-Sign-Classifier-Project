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


[//]: # (Image References)

[image1]: ./outputs/output_8_0.jpg "Visualization"
[image2]: ./outputs/output_13_2.jpg "Random Augmentation"
[image3]: ./outputs/output_16_2.jpg "Preprocess"
[image4]: ./outputs/traffic_sign_1.jpg "Traffic Sign 1"
[image5]: ./outputs/traffic_sign_2.jpg "Traffic Sign 2"
[image6]: ./outputs/traffic_sign_3.jpg "Traffic Sign 3"
[image7]: ./outputs/traffic_sign_4.jpg "Traffic Sign 4"
[image8]: ./outputs/traffic_sign_5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
You're reading it! and here is a link to my [project code](https://github.com/cjpan/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Summary statistics

The code for this step is contained in the 1st and 2nd code cells of the IPython notebook.  

There are data files for training set, validation set and testing set. There is no need to split training set for validation because there are independant validation set.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of test set is 12630.
* The shape of a traffic sign image is 32*32 with 3 channels.
* The number of unique classes/labels in the data set is 43.

#### 2. Exploratory Visualization

The code for this step is contained in the 3rd code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing examples distributions by each label in train, valid and test sets. The examples distributions are similar in training, validation and test sets. However, the distribution in each set are unbalance. There are more examples of some classes than those of some other classes.
Oversampling and undersampling would probably help in this situation.
(I would try it after the project submission.)

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Data Augmentation

The code for this step is contained in the 5th code cell to  of the IPython notebook.

As a first step, I decided to generate additonal data by random translation([-2, 2] pixels in both x and y coodinations), random scaling([-0.9, 1.1] ratio) and random rotation([-15, 15] degrees) to reduce overfitting and make the model more robust to the deformations.

Here is an example of a traffic sign image and a series of augmentation.

![alt text][image2]

#### 2.Data Preprocess
The code for this step is contained from the 5th code cell to 8th code cell of the IPython notebook.

Second, I convert the color space of datas from original RGB to YUV. And then I do a contrast normalzation in all YUV channels. The normalization is to nomalize all features into same [0, 1] ranges. I also found normalization in YUV spaces gets better performance in training and test than that in RGB spaces.

Here is an example of an original image and a series of preprocessed images:

![alt text][image3]

My final training set had 139196 number of images totally. My validation set and test set had 4410 and 12630 number of images(unchanged from the original numbers).

At last step, I shuffle the training set and their labels to generate a random order for training. This step is located in the

#### 3.Model Architecture
The code for my final model is located in the 11th cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 YUV image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 kernel size, 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64      									|
|	RELU					|												|
| Max pooling	      	| 2x2 kernel size, 2x2 stride,  outputs 5x5x64 				|
|	Flatten				|	outputs 1600x1											|
| Fully connected	layer 1	| inputs 1600, outputs 480        									|
|	RELU					|												|
| Fully connected	layer 2	| inputs 480, outputs 120        									|
|	RELU					|												|
| Fully connected	layer 3	| inputs 120, outputs 84        									|
|	RELU					|												|
| Fully connected	layer 4	| inputs 84, outputs 43        									|
|	Softmax					|												||

#### 4. Train the model

The code for training the model is located in the 10th to 14th code cells of the ipython notebook.

To train the model, I used an Adam optimizer to optimize with minimum mean of cross entropy. I used learning rate of 0.001 and batch size of 128. I run 20 epochs for the training.

#### 5. Solution

The code for calculating the accuracy of the model is located in the 14th and 15th cells of the Ipython notebook.

My final model results were:
* training set accuracy of 0.995934
* validation set accuracy of 0.975737
* test set accuracy of 0.953

I used Le-Net5 architecture as a starting point. It is a popular network in deep learning and image classification. I tried with graycaling image and input layer with 32*32*1 at first. I found the highest validation accuracy is only around 0.94, a bit underfitting than the later results.

Then I tried to make the network wider with more features and deeper with one more additional FCL. Also I use color image instead of grayscale image, because I think the color in the traffic signs would help in the classification. I used YUV images instead of RGB images, because I suppose that the preprocess in YUV channels may extract more discrete information than that in RGB.

Then the network can generate a better performance of validation accuracy at around 0.97.

In addition, I also tried tanh function as activation, but it does not work better than RELU.

I run the training 20 times to get a relatively more stable result around 0.97.

The training accuracy, validation accuracy and testing accuracy are all above 0.95. The result are not underfitting. While all the accuracy results are close to each other, which means not big overfitting problem.

### Test a Model on New Images

#### 1. New Images Summary

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The 1st image might be difficult to classify because some part is unclear with snow covered.

![alt text][image4]

The 2nd image might be difficult to classify because it has some deformation which may differ from the training sets.

![alt text][image5]

The 3rd image might be difficult to classify because it is not clear when it is resizing to 32*32. The center part may be easily misclassified to other classes.

![alt text][image6]

The 4th image might be difficult to classify because it has some overlapping in the right margin.

![alt text][image7]

The 5th image might be difficult to classify because there are some parts of some other signs in the image. In addition, when it is resized to 32*32, the central part will be quite unclear for classification.

![alt text][image8]

#### 2. Prediction on New Images

The code for making predictions on my final model is located in the 19th and 20th cells of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| General caution      		| General caution    									|
| Turn right ahead   			| Turn right ahead										|
| Road narrows on the right					| Road narrows on the right											|
| Speed limit (120km/h)      		| Speed limit (120km/h)				 				|
| Beware of ice/snow	| Children crossing    							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 95%.

#### 3. Certainty of Predictions

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook.

For the 1st image, the model is relatively sure that this is a General caution sign (probability of 1.0), and the image is a General caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| General caution  									|
| 1.95e-19     				| Traffic signals 										|
| 3.71e-20					| Pedestrians											|
| 4.97e-24	      			| Wild animals crossing					 				|
| 1.41e-25				    | Speed limit (70km/h)     							|

For the 2nd image, the model is relatively sure that this is a 'Turn right ahead' sign (probability of 1.0), and the image is a 'Turn right ahead' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Turn right ahead  									|
| 1.58e-12     				| Ahead only										|
| 1.93e-15					| Keep left											|
| 5.65e-17	      			| Speed limit (100km/h)				 				|
| 3.68e-17				    | No vehicles   							|

For the 3rd image, the model is relatively sure that this is a 'Road narrows on the right' sign (probability of 1.0), and the image is a 'Road narrows on the right' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Road narrows on the right 									|
| 1.97e-10     				| Speed limit (20km/h)										|
| 1.74e-10				| Dangerous curve to the left										|
| 2.91e-11	      			| Beware of ice/snow			 				|
| 3.28e-12				    | Pedestrians   							|


For the 4th image, the model is relatively sure that this is a 'Speed limit (120km/h)' sign (probability of 1.0), and the image is a 'Speed limit (120km/h)' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Speed limit (120km/h)									|
| 6.97e-11     				| Speed limit (100km/h)									|
| 5.34e-11				| Speed limit (20km/h)										|
| 1.44e-13	      			| Speed limit (70km/h)			 				|
| 4.89e-16				    | Speed limit (80km/h)   							|

For the 5th image, the model is relatively sure that this is a 'Children crossing' sign (probability of 0.99), and the image is a 'Beware of ice/snow' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99         			| Children crossing									|
| 1.19e-06     				| Beware of ice/snow								|
| 8.03e-13				| Wild animals crossing										|
| 7.34e-13	      			| Bicycles crossing			 				|
| 8.78e-16				    | General caution  							|
