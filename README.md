# IntriUAP
(2024/12/15)
This is the code for "AAAI25 Data-Free Universal Attack by Exploiting the Intrinsic Vulnerability of Deep Models". Supplementary materials and more detailed code will be gradually released in the future.

## Environment setup
We recommend installing the required packages by running the command:
```sh
pip install -r requirements.txt
```

## Directory structure
```
IntriUAP
|-- ILSVRC2012_val_00007942.JPEG
|-- Intri_Attack
|   |-- Convlutional_DBToeplitz.py
|   |-- __pycache__
|   |-- attack3.py
|   |-- attack3resnet.py
|   |-- batchnorm\ as\ transformation.py
|   |-- demo.ipynb
|   |-- untitled.txt
|   |-- utils.py
|-- data
|   |-- alexnet
|   |-- avgpix
|   |-- googlenet
|   |-- resnet152
|   |-- uaps
|   |-- vgg16
|   `-- vgg19
`-- requirement.txt
```

We have placed the singular values of the linear layers for each model in the corresponding model folder within the data directory.

## Data 
The avgpix directory contains various initialization files, like range prior, gaussion noise, unifom noise.

## Demo
We provided a demo, demo.ipynb, for creating IntriUAP with VGG19 model.

## Singular vectors
We have provided the right singular vector corresponding to the maximum singular value for the linear layers of VGG19. If you need other singular vectors, you can download them from [(https://drive.google.com/file/d/16MK7KnPebBZx5yeN6jqJ49k7VWbEYQPr/view) ](https://drive.google.com/drive/folders/1TaCmJQFJsKHmN9GeLrGclmpv8Uj7GpIe?usp=drive_link)
