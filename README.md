# VQA
The project aims at multi-layered understanding of pictures to allow a multi-perspective study and hence engender a visual question answering system.

## Structure of the project
- The **data** directory contains pre-trained models and weights;
- The **modules** directory contains files for individual detection and classification tasks;
- The **utils** directory contains utilty and helper functions.
- The **DeepRNN** directory contains scripts required for image_captioning from DeepRNN.

## Setup
**Python** 3 is required.
- Clone the repository -

`git lfs clone --recurse-submodules https://github.com/shubham1172/VQA.git`

- Install the dependencies -

`pip install -r requirements.txt`

## Usage

`python3 run.py --path path/to/image`

## Reference

Image captioning : [DeepRNN/image_captioning](https://github.com/DeepRNN/image_captioning)
