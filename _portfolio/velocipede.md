---
title: "Velocipede"
excerpt: "A regressor that predicts the number of bicycle trips starting from each Bixi bicycle station using the geographical coordinates of the station. Using RBF Kernel, station latitude and longitudes were converted to features representing the proximity of each station to the cluster centers identified using KMeans. These proximity features were then used to train a random forest regressor."
header:
  # image: /assets/images/velocipede_dashboard.png
  teaser: assets/images/030_regression_modelling_002.png
tags: [ KMeans Clustering, Random Forest Regression, RBF Kernel, Data Visualization, SQL ]
---

![velocipede_dashboard.png](../assets/images/velocipede_dashboard_.png){:class="img-responsive"}

Dashboard on AWS EC2 instance: [link](http://3.96.175.190:8501)

GitHub repository: [link](https://github.com/andrewyewcy/velocipede)