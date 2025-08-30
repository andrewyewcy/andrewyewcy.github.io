---
title: "Velocipede"
excerpt: "A regressor that predicts the number of bicycle trips starting from each Bixi bicycle station using the geographical coordinates of the station."
header:
  image: /assets/images/velocipede_dashboard_.png
  teaser: assets/images/030_regression_modelling_002_teaser.png
tags: [ KMeans Clustering, Random Forest Regression, RBF Kernel, Data Visualization, SQL ]
---

### Brief Description
A regressor that predicts the number of bicycle trips starting from each Bixi bicycle station using the geographical coordinates of the station. Using RBF Kernel, station latitude and longitudes were converted to features representing the proximity of each station to the cluster centers identified using KMeans. These proximity features were then used to train a random forest regressor.

Dashboard on AWS EC2 instance: [link](http://35.183.35.168:8501)

GitHub repository: [link](https://github.com/andrewyewcy/velocipede)