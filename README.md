# Mask R-CNN Car Detection

<p align="center">
  <img width="645" height="471" src="https://media.discordapp.net/attachments/691412588654886932/1126309040838955008/image.png?width=645&height=471">
</p>


Welcome to the Mask R-CNN Car Detection repository! This repository contains code and resources for training and deploying a Mask R-CNN model for car detection in images.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This was a fun project I undertook in 2020 as an exercise in object detection, and was my final project for the North Carolina School of Science and Math (NCSSM) Online Program's Data Science for Scientist course. I wrote a beginner-friendly paper on the performance of several trained Mask R-CNN models published in NCSSM's in-house student journal [Broad Street Scientific](https://issuu.com/ncssmedu/docs/bss2021/62).

This project aims to detect cars in images using the Mask R-CNN algorithm. Mask R-CNN is a state-of-the-art deep learning model that combines object detection and instance segmentation. It can identify and locate objects in an image with pixel-level accuracy.

This repository provides the necessary code and configuration files to train the Mask R-CNN model on a custom car dataset. It also includes pre-trained weights for car detection in the `models` folder, which can be used for inference on new images.

## Dataset

A labeled set of images (from [this dataset](https://www.kaggle.com/datasets/alincijov/self-driving-cars) by Alan Cijov) has been provided in this repository. You can also use a custom dataset. The dataset should include images and corresponding annotations or masks that indicate the car regions in the images. If you don't have a custom dataset, you can use publicly available car datasets or augment existing datasets with car annotations.

## Installation

To install and set up the Mask R-CNN Car Detection project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/aksharye/mask-rcnn-car-detection.git
   ```

2. Navigate to the project directory:
   ```
   cd mask-rcnn-car-detection
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train and use the Mask R-CNN model for car detection, follow the instructions below:

1. Prepare your dataset: Organize your car dataset with images and corresponding annotations or masks. You can use the default dataset provided in the repository.

2. Configure the model: Update the parameters at the top of the file `train_model.py` to specify the dataset path, model hyperparameters, and training settings.

3. Train the model: Run the training script `train_model.py` to train the Mask R-CNN model on your dataset.

4. Evaluate the model: Use the evaluation script `test_model.py` to assess the performance of the trained model on a validation set or test set.

## Contributing

Contributions to the Mask R-CNN Car Detection project are welcome! If you want to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b my-feature`.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your forked repository.
5. Submit a pull request to the `master` branch of the original repository.
6. Ensure your pull request clearly describes the changes and their purpose.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code in accordance with the terms specified in the license.

---

Thank you for your interest in the Mask R-CNN Car Detection project! If you have any questions or issues, please [create an issue](https://github.com/aksharye/mask-rcnn-car-detection/issues) on this repository.
