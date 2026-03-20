# Abstract
Machine learning is a subset of artificial intelligence that creates new algorithms without explicit programming by recognizing patterns in datasets. It can be applied to problems such as image recognition. Artificial neural networks use layers of neurons to model problems, and the size and other hyperparameters of the network significantly impact the speed and accuracy of the model. For instance, large models can overfit and memorize patterns in a training dataset, which makes them unable to maintain high accuracy on unseen data because they are unable to generalize patterns. This study predicted that larger networks would have an increased accuracy in image classification, until an eventual plateau and decline due to overfitting. This experiment used PyTorch to create feedforward models for MNIST, Fashion-MNIST, and CIFAR-10 using various sizes and hyperparameters. It measured the runtime and accuracy of each model across 50 epochs, iterations through the dataset, to evaluate performance. Results showed that larger models using the Adam optimizer overfit and were unable to generalize patterns from the training data to the testing data. Conversely, larger models using stochastic gradient descent underfit and were unable to effectively recognize patterns in the training data. ANOVA testing provides statistical evidence to support that smaller optimized models perform better than larger models in these datasets, which creates guidelines for future model creation. Analysis also highlighted trends in runtime and overall accuracy relative to other hyperparameters. Future research should experiment with other datasets and problem types, expand the hyperparameter search, and increase the epoch count. 

# Research Poster
[image]

# Repository Code and Reproducibility
### Specifications for precise reproducibility
- geforce game ready driver 581.80
- cuda 13.0
### Code structure
