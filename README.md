# Athena AI ML Exam

This repository attempts to implement a model evaluation tool. This includes determining false positives classifications and the calibration details for the dataset and model, such as expected calibration error, max calibration error and a calibration graph. Currently, the tool only supports the ConvNeXt model offered as part of PyTorch's torchvision models ([See the model details here](https://pytorch.org/vision/main/models/convnext.html)).

## Dependencies

| Dependency   | Tested version |
| ------------ | -------------- |
| Python       | 3.10.8         |
| pip          | 22.2.2         |
| torch        | 1.13           |
| scikit-learn | 1.1.3          |
| matplotlib   | 3.6.2          |
| PyQt5        | 5.15.7         |
| tqdm         | 4.64.1         |

## Setup

1. Install Python 3.10 or above. At least Python 3.10 is recommended, as some type hints may cause issues with older versions.
2. Create and activate a virtual environment using `venv`. [See here for OS specific instructions](https://docs.python.org/3/tutorial/venv.html).
3. Install the requirements using `pip install -r requirements.txt`.
4. Copy dataset to a folder in the root of the project.

## Usage

After setup has completed, run:  
`python test.py <path>`  
or  
`python test.py <path> --model <model>`  
replacing `<path>` with the path to the dataset and `<model>` with the model size (tiny, small, base, large). The model sizes can viewed [here](https://pytorch.org/vision/main/models/convnext.html).

## Limitations

### 1. Lack of paralleled operations

This is evident during inferencing as images are not batched before inferencing. This could be improved by using the `DataLoader` class as part of `torch.utils`. Although there may be other aspects of the tool that could be optimised further, it is most notable during inferencing.

### 2. Poor demonstration due to the use of an untrained model

Since weights are not provided to the model, the model is essentially guessing during inferencing. This results in the confusion matrix having a majority of the predictions be the same and the calibration graph only having data for low confidence predictions. Given the context surrounding the purpose of this tool, it would be beneficial to provide the ability to import weights and/or models to allow for more accurate classifications and greater variance of data for the tool to display.
