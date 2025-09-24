# What Changed and What Could Have Changed? State-Change Counterfactuals for Procedure-Aware Video Representation Learning

Official code of [What Changed and What Could Have Changed? State-Change Counterfactuals for Procedure-Aware Video Representation Learning](https://arxiv.org/abs/2503.21055), ICCV 2025.

![image](https://github.com/HCIS-Lab/state-change-counterfactual-video-pretrain/blob/main/counterfactual.gif)

## TODO

- [ ] Error detection on EgoPER
- [ ] Temporal Action Segmentation on EgoPER

## Installation

To create a conda environment with the required dependencies, run the following command:

```bash
conda env create -f environment.yml
source activate cf
```

## Dataset Preparation

### Pre-training

Ego4D/EgoClip

Please refer to [EgoVLP](https://github.com/showlab/EgoVLP) codebase for data preparation. We use the downsampled and chunked video outputs as the input to our method (output from `utils/video_chunk.py`). For summary sentences, we provide the processed summary and narration hierarchy [here](https://dl.fbaipublicfiles.com/hiervl/summary_clips_hierarchy_full.json). The used `egosummary_full.csv` is available [here](https://dl.fbaipublicfiles.com/hiervl/egosummary_full.csv).

### Downstream tasks

GTEA: Please follow [Bridge-Prompt](https://github.com/ttlmh/Bridge-Prompt) to download the raw video and then extract frames from videos.

EgoPRE: .

EpicKitchen & Charades-Ego: Please refer to [EgoVLP](https://github.com/showlab/EgoVLP) codebase for data preparation.

AE2: Please pre-extract frame features for this task, following [Align-Ego-Exo](https://github.com/zihuixue/AlignEgoExo) for the data split.

## Generate State Changes and Their Counterfactuals with [Llama](llama_script)

Please refer to [Llama 3](https://github.com/meta-llama/llama3) for model weights and installation instructions. We use the following scripts to generate state change and counterfactual descriptions for the entire Ego4D dataset. Please note that you will need to modify the paths to Ego4D's annotation files in the scripts.

```
# clip-level state changes and their counterfactuals
cd llama_script
python clip_level_sc_cf.py

# video-level counterfactuals
cd llama_script
python video_level_cf.py
```

## Generate Text Features with [FLAVA](language_extraction)

To extract clip-level narration features, please run

```python
language_extraction/feature_extractor.py
```

To extract video-level summary features, please run

```python
language_extraction/summary_feature_extractor.py
```

## Pretraining

### Running on SLURM cluster

To run the pretraining on a distributed SLURM system, copy the content of `slurm_scripts` to this level directly and run

```
bash mover_trainer.sh job_name
```

The parameters of the SLURM job can be changed in the trainer.sh script. We use 2 nodes, each with 4 32 GB GPUs. The submit schedule first copies the required scripts to a different folder and then runs it from there. This copying ensures the code can be safely edited while a job is in the SLURM queue.

### Running on a single machine

Please run

```python
torchrun  --nnodes 1 --nproc_per_node 8 --master_port 8081  run/train_egoaggregate.py --config configs/pt/egoaggregation.json
```
## Pretraining Checkpoint

The pretraining checkpoint is available [here](https://drive.google.com/drive/folders/1fNGuHmyzqygvgbvvE07GylB90kt-NlTi).


## Downstream Task Training/Testing

### Temporal Action Segmentation (GTEA)
Step 1: Generate features with the pre-trained video model. 

Please note that you will need to specify the dataset, model name, cofig path in the script and the "save_dir" in ./as_configs/gtea/gtea_exfm.yaml.

```
python extract_frame_features.py
```

Please refer to [Bridge-Prompt](https://github.com/ttlmh/Bridge-Prompt) for more details.

Step 2: Train/test ASFormer based on the features.

```
cd ASFormer
python main.py --feature cf --dataset gtea --split 1/2/3/4
python main.py --action eval --feature cf --dataset gtea --split 1/2/3/4
python eval.py -- result_dir path_to_results --split 1/2/3/4/0
```

Please refer to [ASFormer](ASFormer) for more details.

### Temporal Action Segmentation (EgoPER)

### Error Detection

### [AE2](AE2) Action Phase Recognition

```python
python AE2/evaluation/AE2_phase_cls.py
```

## Zero-Shot Downstream Task Testing

### EpicKitchen-100 Zero-Shot Multi-Instance Retrieval

```python
python downstream_script/test_epic.py
```

### Charades-Ego Zero-Shot Action Classification

```python
python downstream_script/test_charades.py
```

### [AE2](AE2) Zero-Shot Action Phase Frame Retrieval

```python
python AE2/evaluation/AE2_frame_retrieval.py
```

## Citation

If you use our code or method, please cite the following paper:

```bibtek
@InProceedings{counterfactual_ICCV_2025,
    author    = {Kung, Chi-Hsi and Ramirez, Frangil and Ha, Juhyung and Chen, Yi-Ting and Crandall, David and Tsai, Yi-Hsuan},
    title     = {What Changed and What Could Have Changed? State-Change Counterfactuals for Procedure-Aware Video Representation Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
}
```

## Acknowledgement

The pretraining and Chrades-Ego, EPIC-KITCHENS test codebase is based on [EgoVLP](https://github.com/showlab/EgoVLP) and [HierVL](https://github.com/facebookresearch/HierVL). 

The feature extraction code is adapted from [Bridge-Prompt](https://github.com/ttlmh/Bridge-Prompt).

The temporal action segmentation code is adapted from [ASFormer](https://github.com/ChinaYi/ASFormer).

The action phase recognition and frame retrieval code is adapted from [AE2](https://github.com/zihuixue/AlignEgoExo)
