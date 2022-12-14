# Teaching a model car to drive

## About

This project is done as a part of the Machine Learning (MTAT.03.227) 2022 course. The goal of the project is to teach a model car ([Donkeycar](http://docs.donkeycar.com/)) to drive around a city like circuit so that it can:
- choose a driving direction based on a left/right arrow sign;
- avoid all the obstacles (pedestrians, animals etc) on the road;
- give way to an another car on the intersection;
- stop in front of a stop sign without hitting it.

The evaluation of the final model and comparison with the other teams will be made on 25th January 2023, when the model will participate in the [ADL Minicar Challenge](https://docs.google.com/document/d/1lKWmzDgB0UsW0jYLL02xssNvKfSIyu-uawcHl9se4VY/edit).

## Technical details

The following instructions can be used to collect data, train models and communicate with a car.

### Connecting to the car

SSH can be used to establish a connection with the car:

```
ssh pi@[car-ip]
```

### Collecting Data

Frames, that can be used to train a model later on, can be captured with following instruction:

```
python manage.py drive --js
```

The collected frames will be located in an automatically created tub folder inside the data folder.

### Transfering files

The Raspberry PI used by the Donkey car is not powerful enough to train a CNN model, which means that this must be done in a different environment. This means that the image files, used to train the model, must be retrieved from the car and then the created model must be sent back. Although it is possible to move files between the car and the computer with an SD card, we found it convenient to use scp instead.

Copying files from the car (tub file example)

```
scp -r pi@[car-ip]:/home/pi/[car-folder-name]/data/[tub-folder-name] /destination/path
```

Copying files to the car (trained model example)

```
scp [model-name].h5 pi@[car-ip]:/home/pi/[car-folder-name]/models
```

### Training a model 

Collected data can be used to train a model in the car folder with a statement:

```
donkey train -- tub ./data/* --model ./models/[model-name].h5
```

This statement will use all the tub files in the data folder.  
Hyperparameters like epoch, batch size etc can be changed in the config.py file that is located in the root car folder.

### Testing the model

The created model can be used to control the car with a following instruction:

```
python manage.py drive --model ./models/[mode-name].h5
```

## Steps so far

We have collected over 100K frames so far with varying speeds, obstacle positions and lighting conditions. The images can be found in the [Google Drive](https://drive.google.com/drive/folders/1s1fuojH4sHv6buKUtdfz4J60KtZhVlmV) folder. The resolution of each image is 120x160.

We have also used augmentation techniques in order to increase the amount of pictures. The first augmentation that we have used is changing the brightness of the images. The track is located near a big window, which means that every time the lighting conditions are a little bit different, so the brightness augmentation should make the model work better in different lighting conditions.  

Original image:  
![Original image](./readme-images/brightness_original.jpg)  

Augmented image:  
![Augmented image](./readme-images/brightness1.jpg)

In addition to brightness augmentation we have also created adversarial examples (we got the idea from a recently defended [masters thesis](https://comserv.cs.ut.ee/ati_thesis/datasheet.php?id=74730&year=2022)), which should make the model more resilient to image noise and also improve driving in the changing lighting conditions. Adversarial examples are images on which the pixels are changed in a way, which causes the model predict wrongly. In the case of model car the wrong prediction would be a wrong driving speed or a wrong driving direction. We used method that is somewhat similar to the Fast Gradient Sign Method. The approached used by us works as follows:
- take an image and use our model to predict the car turning angle;
- mean squared loss is calculated by using the prediction and the desired output;
- input image gradients are calculated;
- apply L2 norm to the gradients;
- multiply gradients by learning rate and subtract results from the input image;
- repeat until necessary output value is achieved;

Original image:  
![Original image](./readme-images/original.jpg)  

Augmented image:  
![Augmented image](./readme-images/adversarial.jpg)

Images were then used to train models with different hyperparameters. We also tried categorial approach instead of regression, but this did not work very well because it made the car drive very unsmoothly (constantly correcting its turning radius, even on a straight).

## Results

The best models so far have been able to drive around the track without hitting any obstacles and stop before stop sign. The most limiting factor, when training models, has been the computing power so far, which means that we can not use all the images at the same time, but we have to choose which ones to use as the training time increases greatly as the number of images grows. It takes roughly 5-6 hours to train a model with 100K images and 25 epochs on an RTX 2060 GPU.

[Video of the car driving](https://drive.google.com/file/d/1e5_91DxPOrO1kGj9GawPWKugJTA5syZL/view?usp=sharing)

There are times where the model struggles to avoid all of the obstacles. This is caused by the position of the obstacles, the field-of-view of the camera and the turning radius of the car. Below are some examples of the more difficult obstacle sequences (obstacles are red dots in the pictures).

Example 1:  
![Example 1](./readme-images/fail1.png)

Also video example of the situation described above: [Video of example 1](https://drive.google.com/file/d/1uclQEHKjTmWkC09ThNpb38wZo-fYlblL/view?usp=sharing)

Example 2:  
![Example 2](./readme-images/fail2.png)

Example 3:  
![Example 3](./readme-images/fail3.png)

## Unsolved problems

There are also some unsolved problems. First, it seems that the resolution 120x160 is not good enough to detect the arrow sign, which tells the car whether it should drive left or right, early enough so that the car is able to make the turn successfully (turn radius of the car is quite poor and it can not be changed). This means that image resolution must be increased. One of the major drawbacks of changing the resolution, is the fact that the original images must be recaptured, which is very time consuming.

Arrow sign left:  
![Left arrow](./readme-images/arrow_sign1.jpg)

Arrow sign right:  
![Right arrow](./readme-images/arrow_sign2.jpg)

The second issue is the fact that we have not used cropping so far, but it can be used to increase the models performance greatly. The saliency map below shows that the model focuses too much on the things that are located outside of the track. By cropping the top half of the image, it is possible to remove these distractions.

Image:  
![Original image](./readme-images/original.jpg)  

Saliency map:  
![Saliency map](./readme-images/saliency.PNG)
