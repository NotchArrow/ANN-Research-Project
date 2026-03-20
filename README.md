# Abstract
Machine learning is a subset of artificial intelligence that creates new algorithms without explicit programming by recognizing patterns in datasets. It can be applied to problems such as image recognition. Artificial neural networks use layers of neurons to model problems, and the size and other hyperparameters of the network significantly impact the speed and accuracy of the model. For instance, large models can overfit and memorize patterns in a training dataset, which makes them unable to maintain high accuracy on unseen data because they are unable to generalize patterns. This study predicted that larger networks would have an increased accuracy in image classification, until an eventual plateau and decline due to overfitting. This experiment used PyTorch to create feedforward models for MNIST, Fashion-MNIST, and CIFAR-10 using various sizes and hyperparameters. It measured the runtime and accuracy of each model across 50 epochs, iterations through the dataset, to evaluate performance. Results showed that larger models using the Adam optimizer overfit and were unable to generalize patterns from the training data to the testing data. Conversely, larger models using stochastic gradient descent underfit and were unable to effectively recognize patterns in the training data. ANOVA testing provides statistical evidence to support that smaller optimized models perform better than larger models in these datasets, which creates guidelines for future model creation. Analysis also highlighted trends in runtime and overall accuracy relative to other hyperparameters. Future research should experiment with other datasets and problem types, expand the hyperparameter search, and increase the epoch count. 

# Research Poster
<img width="4608" height="3456" alt="Research Poster" src="https://github.com/user-attachments/assets/22efa621-8676-4619-a693-7d372116daba" />


# Repository Code and Reproducibility
### Specifications for precise reproducibility
- 11th Gen Intel i5-11400F @ 2.60GHz
- NIVDIA GeForce RTX 4060 8GB
  - Geforce Game Ready Driver 581.80
  - CUDA 13.0
- Python 3.13.3
- Package version from the [requirements.txt](https://github.com/NotchArrow/ANN-Research-Project/blob/master/requirements.txt)
### Code structure
While all of the actual model generation and data logging is in the [Main.py](https://github.com/NotchArrow/ANN-Research-Project/blob/master/Main.py) file, researchers looking to expand on the current experimental design and independent variables should focus on the [TrialManager.py](https://github.com/NotchArrow/ANN-Research-Project/blob/master/TrialManager.py) file. It is used to generate the spreadsheet prior to running the main file.

Looking for raw data? The spreadsheet is [here](https://github.com/NotchArrow/ANN-Research-Project/blob/master/trials.csv).
