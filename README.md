# PySSD

Python implementation of Single Shot Multi-box Detector.

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
