# A Mechanistic Analysis of a Transformer Trained on Symbolic Multi-Step Reasoning Task

This is the official implementation of [A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task](https://arxiv.org/abs/2402.11917).

#### Usage

##### 1. Dependencies
To install dependencies:
```setup
conda env update --file environment.yml
```

##### 2. Training and Evaluation Code
To train a model from scratch or continue the training, use `training.py`. We provide functions that have been used for aanay in `src/utils.py`.

##### 3. Pre-trained Model
The model checkpoint we studied in our work is located in `checkpoint/`. 

##### 4. Replication of Results
The notebook `figures.ipynb` replicates all figures we report in our paper.

#### Citation Information
```bibtex
@misc{brinkmann2024mechanistic,
      title={A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task}, 
      author={Jannik Brinkmann and Abhay Sheshadri and Victor Levoso and Paul Swoboda and Christian Bartelt},
      year={2024},
      eprint={2402.11917},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```