![GitHub Repo stars](https://img.shields.io/github/stars/Sc1anso/IoT-activity-recognition)
![GitHub](https://img.shields.io/github/license/Sc1anso/IoT-activity-recognition)



# IoT-activity-recognition
This is the work of my master degree thesis, it's a comparison between the performance obtained with supervised and unsupervised techniques for the Human Activity Recognition problem

## The project:
The following thesis project, which is based on the following [work](https://ieeexplore.ieee.org/document/9217830), shows a comparison between the most common supervised techniques in the literature for Human Activity Recognition using data from IMU sensors and two unsupervised techniques, namely Self-Organizing Map and K-Means.
The algorithms were tested on two different datasets:
 1) [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) - Dataset provided by the University of California Irvine;
 2) Dataset acquired personally following the guidelines of the previous one but acquiring the data directly from the IoT device used for the experiment.
On the data in both datasets, further analysis was performed on the 561 features considered for the unsupervised techniques in an attempt to optimise everything. This analysis is called ANOVA F and consists of calculating a value per class for each feature that tells us how relevant that feature is to that class, this type of analysis was presented in the following [work](https://ieeexplore.ieee.org/document/8215500).

The results obtained and shown below show that the Self-Organizing Map manages to achieve performances very close to those obtained by the supervised techniques, whereas the results are worse when it comes to K-Means.
Once the performance was compared, an attempt was made to implement the best unsupervised technique (SOM) on an IoT device, namely an M5Stack Gray.

## Structure:
- UCI-HAR Dataset: directory containing the UCI dataset;
- Dataset tesi: directory containing the dataset built using the M5Stack Gray device (zipped using WINRAR);
- plots: directory containing all the plots generated during the execution of the experiment on the dataset we have constructed;
- UCI plots: directory containing all the graphs generated during the e execution of the experiment on the UCI dataset;
- np arr: directory containing some saved data structures in .npy format useful for exporting models. In addition to .npy it is possible to find, in particular for the self-organising map, two .txt files that are those that will be actually exported on the microcontroller and they are:
    - map lst... .txt: contains the neuron-label association for the parameters described in the file name;
    - weights lst... .txt: contains the weights of the model to be exported.
    A version for the UCI dataset also exists for this directory;
- som models: directory containing saved models already trained for unsupervised techniques.
A version for the UCI dataset also exists for this directory;
- weights: directory containing saved weights from supervised techniques;
- analyze_data.py: Python script that takes care of reading the data acquired from the micro-controller and proceeds to create the dataset;
- main.py Python script that consists of the core of the experiment. At contained within it is the code relating to the training and testing of all models and the code concerning the export of the files needed for execution on the edge device. It accepts the following input parameters (in order):   
    - "s" or "u" respectively to perform supervised or unsupervised.
    - "bal" or "no-bal" respectively to perform the study on data balanced or unbalanced data;
    - "our" or "uci" respectively to perform the analysis on our dataset or on the uci dataset. IMPORTANT: if you perform the supervised analysis this should be entered as the third parameter, otherwise as the fifth;
    - "kmeans" or "som" respectively to perform the unsupervised analysis with K-Means or SOM;
    - 'avg', 'min' or 'avgmin' respectively, for ANOVA analysis, to do only the analysis by choosing the mean of the variances per class, only by choosing the minimum of the variances per class, or both analyses;
    - "our" or "uci" to be entered now if the first parameter is "u";
    - "y" or "n" respectively to tell the program whether or not it should save, during execution, all plots and useful data structures to be exported;
- server.py: code to run in a python server to pre-process the data acquired with the M5Stack device for SOM testing;
- main.cpp: M5Stack code.
 
 NOTE: All directories that are not in the repository will be generated at runtime.

 ## Dependencies:
Run:
- install_requirements.bat for Windows;
- install_requirements.sh for Linux;

## Execution instructions

To run supervised tests:

```bash
  python main.py s no-bal our y
```
To run unsupervised tests:

```bash
  python main.py u bal som avgmin uci y
```
To run SOM with custom parameters:

```bash
  python main.py u bal som avgmin uci y 10 50 5 5
```

The last four parameters are, respectively:
- min dimension of the Self-Organizing Map;
- max dimension of the Self-Organizing Map;
- SOM size increase with each test;
- training iterations.


## Results:
![Supervised](https://github.com/Sc1anso/IoT-activity-recognition/blob/main/imgs/diap1.PNG)
![ANOVAF](https://github.com/Sc1anso/IoT-activity-recognition/blob/main/imgs/diap2.PNG)
![KMEANS](https://github.com/Sc1anso/IoT-activity-recognition/blob/main/imgs/diap3.PNG)
![SOM](https://github.com/Sc1anso/IoT-activity-recognition/blob/main/imgs/diap4.PNG)

## TODO:
Fix local data pre-processing in main.cpp.
