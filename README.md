# Model-Comparisons


## The Data
The MNIST Fashion database (https://github.com/zalandoresearch/fashion-mnist) collected a large number of images for different types of apparel. Each image is divided into small squares called pixels of equal area. Within each pixel, a brightness measurement was recorded in grayscale. The brightness values range from 0 (white) to 255 (black). The original data set divided each image into 784 (28 by 28) pixels. 

To facilitate more tractable computations, we have condensed these data into 49 pixels (7 by 7) per image.

For each picture, the first 7 pixels represent the top row, the next 7 pixels form the second row, etc.

The assignment provides the following files:

Training Set: MNIST-fashion training set-49.csv. This file contains 60,000 rows of data.
Testing Set: MNIST-fashion testing set-49.csv. This file contains 10,000 rows of data.
Each file includes the following columns:

label: This provides the type of fashionable product shown in the image.

pixel1, pixel2, …, pixel49: These columns provide the grayscale measurement for the 49 pixels of the image.

## A Practical Machine Learning Challenge
What are the best machine learning models for classifying the labels of the testing set based upon the data of the training set? How small of a sample size do you need to generate the “best” predictions? How long does the computer need to run to obtain good results? To balance these competing goals, we will introduce an overall scoring function for the quality of a classification.

Points = 0.25 * A + 0.25 * B + 0.5 * C

where

A is the proportion of the training rows that is utilized in the model. For instance, if you use 30,000 of the 60,000 rows, then A = 30,000 / 60,000 = 0.5;

B = \(\min\left(1, \frac{X}{60}\right)\), where X is the running time of the selected algorithm in seconds. Algorithms that take at least 1 minute to run will have the value B = 1, which incurs the full run-time penalty.

C is the proportion of the predictions on the testing set that are incorrectly classified. For instance, if 1000 of the 10000 rows are incorrectly classified, then C = 1000 / 10000 = 0.1.

You will create and evaluate different machine learning models on different sample sizes. The quality of different combinations of the models and the sample sizes can be compared based on their Points. The overall goal is to build a classification method that minimizes the value of Points. In this setting, the ideal algorithm would use as little data as possible, implement the computation as quickly as possible, and accurately classify as many items in the testing set as possible. In practice, there are likely to be trade-offs. Part of the challenge will be adapting to the rules of the game to improve your score!