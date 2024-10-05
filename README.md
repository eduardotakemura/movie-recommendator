# Movie Recommendator using Neural Collaborative Filtering

A movie recommendation application that leverages Neural Collaborative Filtering (NCF) to provide personalized movie suggestions based on user ratings and movie metadata.

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Usage and Training](#usage_and_training)
5. [Libraries](#libraries)
6. [Contributing](#contributing)
7. [License](#license)

## Overview

The Movie Recommendator is designed to assist users in discovering movies they may enjoy based on their ratings of a selected set of films.  
The application utilizes deep learning techniques to predict user preferences, taking into account not only the ratings but also important metadata 
such as genres and release years. This leads to more nuanced and accurate recommendations.  
Reference [paper](https://arxiv.org/abs/1708.05031).

## Architecture
![Neural Collaborative Architecture Diagram](https://miro.medium.com/v2/resize:fit:720/format:webp/1*Tqk7Q2q7wsr6MLF8Xl-emg.png)  
*Source: https://towardsdatascience.com/neural-collaborative-filtering-96cef1009401*  

The recommendation model is built upon the Neural Collaborative Filtering (NCF) framework, which combines the strengths of both Generalized Matrix Factorization (GMF) 
and Multi-Layer Perception (MLP) to enhance the prediction of user-item interactions.

1. **Embedding Layers**:
    - The model starts with embedding layers for both users and movies, where each user and movie is represented by a dense vector of fixed size.
    - These embedding vectors capture latent features that reflect user preferences and movie characteristics.

2. **GMF Branch**:
    - The GMF branch performs a simple dot product between the user and movie embeddings.
    - This **linear** interaction captures the collaborative filtering aspect of the recommendation task, where user preferences for movies are modeled based on their historical ratings.

3. **MLP Branch**:
    - The MLP branch consists of multiple fully connected layers that take the concatenated embeddings of the user and movie as input.
    - This branch learns complex, **non-linear** interactions between users and movies, enhancing the model's ability to capture more intricate patterns in the data.

4. **NeuMF (Neural Matrix Factorization)**:
    - The outputs of both the GMF and MLP branches are concatenated and passed through a final fully connected layer and then applied a sigmoid activation, which produces the predicted rating.
    - NeuMF combines the strengths of both branches to create a unified model that improves recommendation accuracy by leveraging both linear and non-linear interactions.

## Installation

Instructions on how to install the project locally.

```bash
# Clone this repository
$ git clone https://github.com/eduardotakemura/movie-recommendator.git

# Go into the repository
$ cd movie-recommendator

# Install dependencies
$ pip install -r requirements.txt
```

## Usage and Training

To run the app using the streamlit frontend, just run:
```bash
# Start the application
$ streamlit run app.py
```
The training was done using Google Colab, so you might need to adjust the following data download to run on Jupyter or other IDE:  
```bash
# External Libraries #
!pip install tensorflow

# Dataset Download #
!wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip ml-latest-small.zip
```
You can also download any other dataset on [MoviesLens web page](https://grouplens.org/datasets/movielens/), and train the model with it.  
All MovieLens datasets are usually structure in the same way, so you can run the script probably without any other adjustment.
I would only say, if you choose a large dataset, that you might need to set a batching loading logic for the dataset, to prevent Colab or your machine to break due RAM usage.  

## Libraries
This project relies most on the following libraries:
- **TensorFlow/Keras**: A powerful open-source library for building and training deep learning models, with Keras providing an intuitive high-level interface.
- **Scikit-learn**: A versatile library for machine learning in Python, offering tools for classification, regression, clustering, and data preprocessing.
- **Numpy**: A fundamental library for numerical computing in Python, providing support for large multi-dimensional arrays and a wide range of mathematical functions.
- **Pandas**: A data manipulation and analysis library that provides flexible data structures (like DataFrames) for handling structured data efficiently.
- **Streamlit**: An open-source framework for creating interactive web applications in Python, designed specifically for data science and machine learning projects.

## Contributing
Contributions are welcome! Please read the contributing guidelines first.

1. Fork the repo
2. Create a new branch (git checkout -b feature/feature-name)
3. Commit your changes (git commit -m 'Add some feature')
4. Push to the branch (git push origin feature/feature-name)
5. Open a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
