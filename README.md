# CS 534 Final Project: LLM-assisted Compositional Image-text Matching

## This repo is built based on Dense and Aligned Captions (DAC) Promote Compositional Reasoning in VL Models - An official repo for the *Spotlight* Neurips 2023 paper :) 

Arxiv: https://arxiv.org/abs/2305.19595

_______________________________

## Environment
```shell script
conda deactivate # deactivate any active environments
conda create -n dac python=3.8.13 # install the conda environment with conda dependencies
conda activate dac # activate the environment
conda install -c conda-forge libjpeg-turbo
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3.1 -c pytorch
pip install -r requirements.txt
```

## Data Preparations and creation 
Please download the data we selected for our work from:
https://drive.google.com/drive/folders/1n3sH7q8F8JMdgfqaEy9FonrNfiFtGKdK?usp=sharing
### Training Data

#### Download CC3M data
Download Conceptual Captions 3M training and validation splits from https://ai.google.com/research/ConceptualCaptions/download  
Download and place in `DAC/CC3M_data/` train_with_cap.csv and val_with_cap.csv from https://drive.google.com/drive/folders/1WosT_kdam1ymWjVSK2ezyydLoqmm0LdX?usp=sharing

We sampled **10,000** images from the original **CC3M** training dataset, each paired with a single quality caption. Using the DAC **framework(Dense Caption)**, we generated **10 dense captions per image** to enrich the training dataset. This structured approach ensures diverse and detailed captions for robust model training.

### Evaluation data
<!---
Prepare vl checklist dataset as described in https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md  
Then move the vl dataset to `DAC/vl_datasets/`  
If you followed the instructions correctly, you should have the following folders inside vl_datasets: **'hake', 'swig', 'vg'**. 

First, navigate to the src directory:
```shell script
cd src
```
--->

In our project, we utilized the **VL\_Checklist** dataset, which is designed to evaluate fine-grained visual-linguistic understanding across categories such as color, size, actions, and spatial relations. Due to time and resource constraints, we sampled a total of **1,000 images**, with 500 images each from the "attributes" and "relations" categories. Specifically, we randomly selected **50–100 images** per subfolder (e.g., `vg/color`, `vaw/color`, `vg/size`, `hack_action`, etc.). The sampled files are stored in `final_sampled_image` and `final_sampled_json`.


#### Create quality captions:

```shell script
mkdir DAC/quality_captions/
python3 training/main.py --create_quality_captions --save_data --batch-size 1 --workers 0
```


#### Create Dense captions:
<!---
```shell script
mkdir DAC/SAM_dense/
python3 training/main.py --create_SAM --save_data --batch-size 1 --workers 0 --model_SAM /path/to/sam_vit_h_4b8939.pth
```
--->
```shell script
mkdir DAC/LLM_dense/
python3 create_LLM_dense.py
```

#### Create Dense captions with minimal hallucination:

```shell script
mkdir LLM_dense_ha/
python3 create_LLM_dense_ha.py
```

<!---
### Evaluation data
Prepare vl checklist dataset as described in https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md  
Then move the vl dataset to `DAC/vl_checklist_images_root_folder/`  
If you followed the instructions correctly, you should have the following folders inside vl_datasets: **'hake', 'swig', 'vg'**. 

prepare aro dataset as described in https://github.com/mertyg/vision-language-models-are-bows
Then move the aro dataset to `DAC/aro/` 
--->

## Train with Quality and Dense data

### Run the training script

The model will be saved in `DAC/Outputs/exp_name/checkpoints`

To train a network with quality captions and:
<!---
* SAM density:
```shell script
python3 training/main.py --epochs 5 --name exp_name --lora 4 --use_only_quality_captions --batch-size 32 --mil_dense_negs --vl_negs --neg_type rand_both --auto_neg_types NOUN ADP ADJ VERB --mil_batch 10 --pretrained openai --mil_dense ../SAM_dense/
```
---> 

* LLM density:
```shell script
python3 training/main.py --epochs 5 --name exp_name --lora 4 --use_only_quality_captions --batch-size 32 --mil_dense_negs --vl_negs --neg_type rand_both --auto_neg_types NOUN ADP ADJ VERB --mil_batch 10 --pretrained openai --mil_dense ../LLM_dense/
```

To train a network with quality captions and:
* LLM density with minimal hallucination:
```shell script
python3 training/main.py --epochs 10 --name exp_name --lora 4 --use_only_quality_captions --batch-size 32 --mil_dense_negs --vl_negs --neg_type rand_both --auto_neg_types NOUN ADP ADJ VERB --mil_batch 10 --pretrained openai --mil_dense ../LLM_dense_ha/
```

## Evaluation
### Run the evaluation script
#### you can download the baseline checkpoints of DAC_SAM and DAC_LLM from here: https://drive.google.com/drive/folders/1DmHeV8oWiMwtkaTH-nruMyjBiuJvcwnv?usp=sharing
<!---
All vl_checklist jsons will be saved in `DAC/eval_jsons/clip/exp_name/` and the result will be printed. 
To prepare the vl checklist evaluate results for the experiment **exp_name** run the following command:
```shell script
mkdir vl_checklist_accuracy_jsons_folder
python3 training/main.py  --lora 4 --pretrained openai --eval_vl_cklist --eval_only --resume /path/to/checkpoint --vl_checklist_images_root_folder DAC/vl_checklist_images_root_folder/
```

To print the aro evaluated results for the experiment **exp_name** run the following command:
```shell script
python3 aro_clip_lora_eval.py  --lora 4 --resume /path/to/checkpoint
```
---> 
