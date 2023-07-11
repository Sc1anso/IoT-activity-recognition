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
