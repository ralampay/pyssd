# PySSD

Python implementation of Single Shot Multi-box Detector.

## Modules

### Training

Train an SSD model based on parameters found in `--config-file` option.

Example Command:

```
python -m pyssd --mode train --config-file test/sample_train_config.json
```

### Detect Image

Detect objects in a given image specified in parameters found in `--config-file` option.

Example Command:

```
python -m pyssd --mode train --config-file test/sample_detect_image_config.json
```

## Training in Google Colab

[https://colab.research.google.com/drive/1n53gw3uNTb-zjHhSm-qQ0xrp7a_bJqZW#scrollTo=KA08HVXyZpHL](https://colab.research.google.com/drive/1n53gw3uNTb-zjHhSm-qQ0xrp7a_bJqZW#scrollTo=KA08HVXyZpHL)

## Development

Install the dependencies:

```
pip install numpy tqdm torch torchvision opencv-python
```

When switching base models see if your initial convolution based model is compatible, make sure to run either of the following:

```
python test.py
```

```
python test_image.py
```

```
python test_train.py
```

## Dependencies

* `numpy`
* `tqdm`
* `torch`
* `torchivision`
* `opencv-python`

**Source:** https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
