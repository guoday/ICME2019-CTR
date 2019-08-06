## Short Video Understanding Challenge

Introduction of ICME2019 Grand Challenge refers to https://biendata.com/competition/icmechallenge2019/

More detail coming soon!



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


