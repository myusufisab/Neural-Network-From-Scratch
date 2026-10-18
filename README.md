# Neural Network Built from Scratch — Iris Classification

A simple feedforward neural network implemented from scratch in Python to classify Iris flowers using their physical measurements. This project explores the fundamentals of neural networks, sigmoid activation, backpropagation, and gradient-based learning without relying on machine learning frameworks.

## Overview

This project implements a basic neural network using Python's built-in `random` and `math` modules. The network learns to distinguish between two Iris flower classes using four input features.

The goal was to understand how a neural network processes input data, generates predictions, calculates errors, and updates its weights and biases during training.

## Features

- Neural network implemented from scratch
- Four input features representing flower measurements
- Four neurons in a single hidden layer
- One output neuron for binary classification
- Sigmoid activation function
- Backpropagation for calculating weight updates
- Gradient-based learning using a learning rate
- Training over 2,500 epochs
- No dedicated machine learning framework required

## Neural Network Architecture

The model uses a simple feedforward architecture:

**4 input features → 4 hidden neurons → 1 output neuron**

### Input Features

The network receives four normalized measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

The supplied dataset contains 30 samples, with 15 examples from each of two flower classes.

### Classification Targets

The output labels are encoded as:

| Target | Flower Class |
|---|---|
| `0` | Iris Setosa |
| `1` | Iris Versicolor |

The output neuron produces a value between 0 and 1 using the sigmoid function. A classification threshold would be needed to convert that output into a final predicted class.

## Technologies Used

- **Python**
- `math` — sigmoid activation and mathematical operations
- `random` — random initialization of weights and biases

No external machine learning libraries are used in the training implementation.

## How It Works

### 1. Weight and Bias Initialization

The network initializes its weights and biases with small random values. These parameters are adjusted during training so the model can learn patterns in the input data.

### 2. Forward Propagation

Each hidden neuron calculates a weighted sum of the four input features, adds its bias, and applies the sigmoid activation function.

The hidden-layer activations are then combined to produce the output prediction.

### 3. Error Calculation

The difference between the target value and the predicted output is calculated:

`error = target - prediction`

The model uses this error to determine how its parameters should change.

### 4. Backpropagation

The output error is propagated backward through the network. The code calculates update signals for the output neuron and hidden neurons using the derivative of the sigmoid function.

### 5. Weight and Bias Updates

The model updates its weights and biases using a learning rate of `0.01`.

This process repeats across the dataset for 2,500 epochs, allowing the network to adjust its parameters over time.

## Getting Started

### Requirements

- Python 3
- No external packages required

### Run the Project

1. Save the Python code in a file named `neural_network.py`.
2. Open a terminal in the directory containing the file.
3. Run:

```bash
python neural_network.py
```

The program prints periodic training predictions and displays a training-complete message when the training loop finishes.

## What I Learned

- How a feedforward neural network processes numerical inputs
- How weights and biases influence predictions
- How sigmoid activation works
- How backpropagation helps calculate parameter updates
- How learning rates affect weight updates
- How neural networks can learn from labelled data
- How to implement the basic mechanics of a neural network without a dedicated ML framework

## Current Limitations

- The dataset contains only 30 samples.
- The model is trained and evaluated on the same samples; no separate test set is implemented.
- The code prints a prediction during training but does not calculate final classification accuracy.
- Training results can vary because the initial weights and biases are random.
- The current implementation is a learning exercise rather than a validated production classifier.

## Future Improvements

- Add a separate training and testing dataset.
- Calculate classification accuracy after training.
- Add a function for making predictions on new flower measurements.
- Set a random seed for reproducible initialization.
- Refactor repeated neuron calculations into reusable functions.
- Experiment with different learning rates and network architectures.

## Project Purpose

This project was created as a learning exercise to explore the mathematical foundations of neural networks and gain practical experience implementing a simple machine learning model using Python.

