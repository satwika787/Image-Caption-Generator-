# Image-Caption-Generator
# Image Captioning with VGG16 and LSTM

This project demonstrates how to build an image captioning model using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The VGG16 model is used as the feature extractor, and an LSTM model is utilized for generating captions.

## Project Overview

The project involves:

1. Extracting image features using a pre-trained VGG16 model.
2. Preprocessing text data using tokenization and padding.
3. Training an LSTM model to generate captions based on image features and textual inputs.
4. Utilizing the Flickr8k dataset for training and evaluation.

## Dataset

The dataset used is the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k), which contains 8,000 images and 40,000 captions.

### Downloading the Dataset

The dataset can be downloaded from Kaggle. Follow these steps:

1. Set up Kaggle API credentials by uploading your `kaggle.json` file.
2. Move the `kaggle.json` file to the appropriate directory:
   ```bash
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
Download the dataset:
bash
Copy code
kaggle datasets download -d adityajn105/flickr8k
Extract the dataset:
bash
Copy code
unzip flickr8k.zip
Installation
To run this project, you need to have the following dependencies installed:

```bash
pip install keras tensorflow numpy tqdm kaggle

## Project Structure
The key components of this project are:

Feature Extraction: Extracting features from images using the pre-trained VGG16 model.
Text Preprocessing: Tokenizing and padding captions using the Keras Tokenizer and pad_sequences.
Model Training: Training an LSTM model that combines image features with text data.
Caption Generation: Generating captions for new images based on trained models.

### Key Libraries Used
tensorflow
keras
numpy
tqdm
kaggle
#### How to Run
Install all required libraries.
Download and extract the dataset.
Upload your kaggle.json file using:
  ```bash
  from google.colab import files
  files.upload()
  Move the kaggle.json file as described in the Installation section.
Run the code to extract features, preprocess data, and train the model.

### Usage
You can customize the code for different image datasets or experiment with various pre-trained models. The project is modular, allowing easy integration of other feature extractors or different text models.


##### License
This project is licensed under the MIT License - see the LICENSE file for details.


This README covers the project overview, dataset instructions, installation requirements, project st
