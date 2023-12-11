# OS-SAR (Navigating Open Set Scenarios for Skeleton-based Action Recognition)
![](https://img.shields.io/badge/version-1.0.1-blue)
[![arxiv badge](https://img.shields.io/badge/arxiv-TODO-red)](TODO)
[![AAAI](https://img.shields.io/badge/AAAI-2024-%23f1592a?labelColor=%23003973&color=%23be1c1a)](TODO)

>In real-world scenarios, human actions often fall outside the distribution of training data, making it crucial for models to recognize known actions and reject unknown ones. However, using pure skeleton data in such open-set conditions poses challenges due to the lack of visual background cues and the distinct sparse structure of body pose sequences. In this paper, we tackle the unexplored Open-Set Skeleton-based Action Recognition (OS-SAR) task and formalize the benchmark on three skeleton-based datasets. We assess the performance of seven established open-set approaches on our task and identify their limits and critical generalization issues when dealing with skeleton information.To address these challenges, we propose a distance-based cross-modality ensemble method that leverages the cross-modal alignment of skeleton joints, bones, and velocities to achieve superior open-set recognition performance. We refer to the key idea as CrossMax - an approach that utilizes a novel cross-modality mean max discrepancy suppression mechanism to align latent spaces during training and a cross-modality distance-based logits refinement method during testing. CrossMax outperforms existing approaches and consistently yields state-of-the-art results across all datasets and backbones. We will release the benchmark, code, and models to the community.

- Due to the **```page and format restrictions```** set by AAAI publications, we have omitted some details and appendix content. For the complete version of the paper, including the **```selection of prompts```** and **```experiment details```**, please refer to our [arXiv version](TODO).

## 🤖 Model Architecture
![Model_architecture](https://github.com/KPeng9510/OS-SAR/blob/main/figure/main.png)

## 📚 Dataset Download
- NTU 60
- NTU 120
- ToyotaSmartHome
#### NTU RGB+D 60 and 120
1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`
### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```
#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```
# Training & Testing

### Training

- Change the config file depending on what you want.

```
# Example: training CTRGCN on NTU RGB+D 120 cross subject with GPU 0
python main.py --config config/nturgbd120-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/ctrgcn --device 0
# Example: training provided baseline on NTU RGB+D 120 cross subject
python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.baseline.Model--work-dir work_dir/ntu120/csub/baseline --device 0
```

- To train model on NTU RGB+D 60/120 with bone or motion modalities, setting `bone` or `vel` arguments in the config file `default.yaml` or in the command line.

```
# Example: training CTRGCN on NTU RGB+D 120 cross subject under bone modality
python main.py --config config/nturgbd120-cross-subject/default.yaml --train_feeder_args bone=True --test_feeder_args bone=True --work-dir work_dir/ntu120/csub/ctrgcn_bone --device 0
```
## 📕 Installation

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX

- We provide the dependency file of our experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running `pip install -r requirements.txt `
- Run `pip install -e torchlight` 

## 🤝 Cite:
Please consider citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

```bigquery
TODO
```
