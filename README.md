# UDPCS
UDPCS: Unsupervised Domain Adaptation refinement using Pseudo-Candidate Sets 

### Setting Up the Project

Follow these instructions to set up the project environment and data.

#### 1. Create a New Conda Environment

To isolate your project dependencies, create a new Conda environment. Open your terminal and run the following command:

```bash
conda create --name udpcs python=3.10.13
```
#### 2. Activate the Conda Environment

```bash
conda activate udpcs

```
#### 3. Install Required Packages

```bash
pip install -r requirements.txt
```
#### 4. Download the VisDA Dataset
Download the VisDA Dataset from the official website (https://github.com/VisionLearningGroup/taskcv-2017-public (classification track)). 
Once downloaded, organize the dataset by placing it in the /data/visda folder. The directory structure should resemble this:

```bash
data/
│
├── train/
│   ├── Aeroplane/
│   ├── ...
│
├── validation/
│   ├── ...
│
└── imagelist/
    ├── ...
```
#### 5. 
First download the trained UDA model (MDD) weight file using the following steps.

a) Navigate to logs/VisDA2017/checkpoints/weights

b) Execute the command "cat model_chunk_* > best.pth.gz"

c) Execute the command "gunzip best.pth.gz"

Place this unzipped .pth file in the directory logs/VisDA2017/checkpoints/

Ensure you do not make any modifications to the "imagelist" folder. This folder contains paths to images across all domains.
### 6.
### Train

- [x] UDPCS+MDD on `VisDA` dataset:
     The following command is provided for in the mdd.sh file.

  ```bash
   CUDA_VISIBLE_DEVICES=3 python3 UDPCS_MDD.py /data/visda -d VisDA2017 -s Synthetic -t Real -a resnet101 --epochs 30 --bottleneck-dim 1024 --seed 0 --train-resizing cen.crop --per-class-eval -b 36 --log logs/VisDA2017
  ```
  (or)
Execute the mdd.sh file as shown below

```bash
bash mdd.sh
```
### Output
 - [x] Upon the completion of training for 30 epochs, the metric 'test_acc1'is shown as an indicator of test accuracy after the refinement of MDD model with UDPCS technique.
