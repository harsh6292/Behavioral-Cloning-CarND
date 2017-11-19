## **Behavioral Cloning**


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sample_images/model_summary.png "Model Visualization"
[image2]: .//sample_images/center_2017_11_17_20_02_06_742.jpg "Center Image"
[image3]: .//sample_images/center_2017_11_17_20_02_25_717.jpg "Center Image 1"
[image4]: .//sample_images/center_2017_11_17_20_02_26_722.jpg "Center Image 2"
[image5]: .//sample_images/center_2017_11_17_20_02_27_753.jpg "Center Image 3"
[image6]: .//sample_images/center_2017_11_17_20_02_28_719.jpg "Center Image 4"
[image7]: .//sample_images/center_2017_11_17_20_04_19_244.jpg "Center Image pt2 1"
[image8]: .//sample_images/center_2017_11_17_20_04_21_398.jpg "Center Image pt2 2"

[image9]: .//sample_images/center_2017_11_17_19_56_38_414.jpg "Original Image"
[image10]: .//sample_images/flipped_image.jpg "Flipped Image"


---
#### Files Submitted

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* report.md summarizing the results

To run the car in the simulator autonomously around the track, below command can be used to give steering angles to the car:

```sh
python drive.py model.h5
```

Also, the model.py file contains the code written to extract the training images from above zip, pre-process the images and pass it through a Keras model to be trained and validated.


#### Model Architecture and Training Strategy

##### Model architecture
The keras model I created consisted of total of 9 layers.

* The first layer is the Cropping2D layer through which I cropped the image so that only the important parts of the image are seen by the model (The road, side lines etc). This eliminated the hood of the car and any scenery or sky from the image which is not relevant for the model.
(modely.py line 186)

* The second layer consisted of Lambda normalization layer. This layer helps normalize the image which might have extreme pixel values at certain points.
(model.py line 189)

* The third, fourth and fifth layers are the convolution layers with increasing filter size of 8, 12 and 16 respectively. The filter size varied from 5x5 to 3x3 in the end. These layers use the LeakyReLU activation function to introduce nonlinearity with little weights given to output values which does not pass threshold.
(model.py lines 191-222)

* The sixth layer is the flatten layer to convert convolution inputs into one single output.
(model.py line 229)

* After this, three Dense layers or fully connected layers were added.
(model.py lines 232-240)


##### Attempts to reduce overfitting in the model
The model made use of flipped images in order to generalize better.

I augmented my own training dataset with Udacity provided dataset in order to reduce overfitting and generalize better.

I also used LeakyReLU and SpatialDropout layers in the model to reduce overfitting.

I then ran the saved model.h5 file to be used with simulator to verify that the car is well inside the lanes and even if it touches the lane, the model brings it back to center of lane.


##### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 247).


##### Appropriate training data

I first used the Udacity provided training data to see how the car behaves on track. I then augmented my own training data with Udacity data to provide more examples to model for situations involving when car goes off track.

I also used combination of left and right images along with center to keep the car in lane and recover back when it goes out of the way.

For details about how I created the training data, see the next section.

---
##### Solution Design Approach

I followed the Udacity guidelines and videos to construct an initial model with just one dense layer as the output layer. I then added two convolution layers of size 5x5 each with filter depth of 16 each. To capture more features I added two dense layers too. The model performed well on straight roads, but was way off the track on curves.

This model performed good on training set, but on the validation set the validation loss was much higher than the training loss. On running the car in simulator with this model proved right in sense that car was making wrong turns at every point.

I then added one more convolution layer after the two layers to capture the high-level features in the image. This layer consists of filter depth 16 with filter size of 3x3. I changed the earlier convolution layers to have filter depth of 8 and 12 respectively with filter size of 5x5 each.

This model was appropriate as it did not had so many features to work on but simultaneously complex enough to capture the most important features in the image.

The validation loss now decreased considerably and performed well also in the simulator with some minor errors. I added more training data and data augmentation which improved the results in such a way that the car was now able to drive by itself with no errors.

Lastly, I implemented LeakyReLU and SpatialDropout layers to further reduce overfitting and generalize better. This is evident when running the car in simulator in autonomous mode.


##### Final Model Architecture

The final model architecture (model.py lines 170-244) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][image1]

##### Creation of the Training Set & Training Process

To capture good driving behavior, I first captured single lap on track one using center lane driving.

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to steer back to center of the lane when it is about to go off track.

These images show what a recovery looks like starting from left side of the track (car is on the track line) and then moving back to the center:

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]

Then I started driving the car in reverse direction as the track one is left-steering bias which model might pick up and not generalize well. I recorded one lap in reverse direction with in between going off track and steering back to center.


I also recorded some portion of data on second track so that model does not learn bias towards track one only.

![alt text][image7]

![alt text][image8]


To augment the data set, I used the left and right image points too so that model can learn to steer back into position if it gets too much close to left or right track.
(model.py lines 70-107)


I also flipped images and angles thinking that this would eliminate left-steering bias of track one. For example, here is an image that has been flipped:

![alt text][image9]

![alt text][image10]


After the collection process, I had total of 25,422 images (including augmented images) for the model to train upon. I then preprocessed this data by first cropping the image and keeping only the relevant parts of the image. About 50 pixels from the top of the image were neglected and similarly 20 pixels from bottom were removed.

I used Adam optimizer to train the model so that manually training the learning rate was not necessary.
(model.py line 247)

The ideal number of epochs (10) was found after running several runs.
I used the ModelCheckpoint method to automatically save the best model with least validation loss.
(model.py line 251-252)

I also used the generators for both training and validation sets to reduce memory consumption and process images in batches only.
(model.py line 138-161, 261)

The loss first decreases and then increases by epoch 4 and then finally decreases as it reaches epoch 8.

I used the Keras built-in fit_generator method to train on the data and used generators to return batch size of 32 shuffled images.

I also used train_test_split method from sklearn to split the training data and put aside 20% of the training data as validation set.
