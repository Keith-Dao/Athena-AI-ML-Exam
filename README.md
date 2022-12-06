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
