# T2C instructions

To use Talk2Car with Vilbert please follow these instructions.

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git
cd vilbert-multi-task
pip install -r requirements.txt
```

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

3. Install apex, follows https://github.com/NVIDIA/apex

4. Install this codebase as a package in this environment.
```text
python setup.py develop
```

Next we need to prepare the data.
In the folder ``data`` please create a folder `t2c` and drop all data from 
`https://github.com/talk2car/Talk2Car/tree/master/c4av_model/data` in this folder, together with `https://github.com/talk2car/Talk2Car/blob/master/c4av_model/utils/vocabulary.txt`
Also download the images from [here](https://drive.google.com/file/d/1bhcdej7IFj5GqfvXGrHGPk2Knxe77pek/view).

Next, we are going to extract the features

## Extracting features

1. Install [`vqa-maskrcnn-benchmark`](https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark) repository and download the model and config. 

```text
cd data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
```

```text
python script/t2c_extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --output_folder <path_to_output_extracted_features>
```

```text
python script/gt_t2c_extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --output_folder <path_to_output_extracted_features>
```

4. Convert the extracted images to an LMDB file

```text
python script/convert_to_lmdb.py --features_dir <path_to_extracted_features> --lmdb_file <path_to_output_lmdb_file>
```
This should be enough to run the model

### Fine-tune from Multi-task trained model

```
python train_tasks.py --bert_model bert-base-uncased --from_pretrained <multi_task_model_path> --config_file config/bert_base_6layer_6conect.json --tasks 19 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name finetune_from_multi_task_model
```
 