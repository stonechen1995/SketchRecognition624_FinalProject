# SketchRecognition624_FinalProject
 The aim of this project is to develop a data augmentation algorithm that simulates the random jitter that occurs during handwriting. This algorithm is  used to generate new, augmented data based on a small subset of the original data. The goal of this work is to improve the generalization performance of machine learning models by providing them with additional, artificially generated training examples.
 
This project has following featrues:
- Extract skeletons of the input images
- Generate new skeleton images based on input images with random jitters
- Reconstruct new handwriting images from skeletons
- Using ResNet-152 non-pretrained model to do training and testing

## Installation
To install the project, following the steps:
1. Install required packages running `pip install -r requirements.txt`
2. Clone this repository and navigate to the project directory
3. Download dataset from https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset to the project directory

## Usage
To run the project, go to the project directory and run
```
python deformation.py
```
And a folder called `new_image` will be created at the same directory. The newly generated images are saved in the `new_image` folder.

To train ResNet-152 model by the new image dataset, following the steps:
1. Upload the `new_image` folder to your root directory of Google Drive
2. Go to the project directory and run ```python resnet_152.py```

## Resources
- [OpenCV Tutorial](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Pytorch Tutorial](https://pytorch.org/tutorials/)
- [Scikit-learn Tutorial](https://scikit-learn.org/stable/tutorial/index.html)

## Contact
For questions or support, please contact us at [ruichenn98@gmail.com].


