# WeatherGFM
The official code for paper "WeatherGFM: Learning a Weather Generalist Foundation Model via In-context Learning"

# 1. Download datasets
Download raw sevir dataset from [sevir](https://sevir.mit.edu/sevir-dataset), era5_128X256 from [era5](https://mediatum.ub.tum.de/1524895)

# 2. Process datasets

Process raw sevir, run:

```
python data/upload_events.py --raw_sevir_dataset_path --new_data_save_path
```

Process raw era5, run:

```
python data/upload_era5.py
```

# 3. Download pretrained models

# 4. Test

a. Different task using different prompt, we test up to 25 different tasks:
```
bash run_test.sh
```
b. According the results from step a, testing metrics:
```
python test_all_metrices.py
```

# 5. Continue training/Finetune
You can also use custom dataset to train your own model, just run this command:

```
bash run_train_large_finetune.sh
```
