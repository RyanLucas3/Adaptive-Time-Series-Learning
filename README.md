# Adaptive-Time-Series-Learning

This repository contains code that I originally wrote in https://github.com/parleyyang/OPRG1. For the purposes of seperating it out from things I didn't write and
since that other repository contains other less relevant things, this repository contains my code base for implementing Adaptive Time Series Learning algorithms.

The code was written for this publication https://arxiv.org/pdf/2110.11156.pdf, and closely matches the methodology you'll find outlined there. The method is a time series statistical learning algorithm, called Adaptive Learning, that is capable of handling model selection, out-of-sample forecasting and interpretation in a noisy environment such as a macroeconomic or financial time series. This code covers two main algorithm types - Dynamic Model Selection (DMS) and Ensembling. DMS is a form of Adaptive Learning that provides a single choice for the forecasting model from the set of candidate models. Ensemble Adaptive Learning is a way of ensembling the candidate model forecasts, while also accounting for model quality through the core statistical learning algorithm.

The code also covers various types of local loss functions, which fall under four main types:

  1. Single-Valued Norm. Involves measuring loss as a simple MAE or MSE measure and choosing the best model as the one minimising either quantity.
  2. Multi-Valued Norm. When forecasts are made on a longer forecast horizon, one can measure the loss associated with "intervening" forecasts - those that lie between the current time point and the eventual forecast horizon.
  3. Model-Comparison. When a best model is suspected ex-ante, one can choose to evaluate and penalise the loss of other models relative to this model.
  4. Huber Loss. Huber loss is motivated by outlier detection. When looking at previous loss values, one can check whether a model had particularly bad, outlying forecasts. These are penalised more heavily under this loss function, and hence we reward models for stability.

The method also offers interpretability on variable importance and underlying model selection, making it a "white-box" machine learning method, as opposed to black-box neural networks, etc. This opens up a suite of cool interpretability applications, from general indications of variable importance to time-varying interpretations and case studies of important events. The below provides an example of an illustration I created that demonstrates the interpretability benefits of Adaptive Learning models during the 2020 financial crisis.

![Interpretation](https://user-images.githubusercontent.com/55145311/148679369-fbeb942a-0df3-4cce-9a98-0671d583d3c6.png)
