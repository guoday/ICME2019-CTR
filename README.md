## Introduction

Implement of the paper ["Multi-modal Representation Learning for Short Video Understanding and Recommendation"](https://ieeexplore.ieee.org/document/8795067)

## Short Video Understanding Challenge

Introduction of ICME2019 Grand Challenge refers to the [website](https://biendata.com/competition/icmechallenge2019/)


### 1. Requirement

- scikit-learn

- tqdm

- pandas

- numpy

- scipy

- tensorFlow=1.12.0 (≥1.4 and ≠1.5 or 1.6)

- 128G memory and 1 GPU

  

### 2. Download Dataset

```shell
cd data
bash download.sh
cd ..
```

### 3. Preprocess data

```
python preprocess.py
```

### 4. Extract features

```shell
python extract_features.py
```

### 5.Convert format of data

```shell
python convert_format.py
```

### 6. Train the model

```shell
python run.py
```


