## Multimodal Final Project<br><sub>[Boosting Text-to-Image Diffusion Models via Initial Noise Optimization](https://arxiv.org/abs/2404.04650)</sub>

## Getting started

**Python libraries:** You can use the following commands to create and activate your InitNO Python environment:

```.bash
# Create conda environment
conda env create -f environment.yaml
# Activate conda environment
conda activate initno_env
```

**Generating images:** Run the following command to generate images.
```.bash
python run_sd_initno.py
```

You can specify the following arguments in `run_sd_initno.py`:

* `SEEDS`: a list of random seeds
* `PROMPT`: text prompt for image generation
* `token_indices`: a list of target token indices
* `result_root`: path to save generated results

For **Our Improvements**, we provide the following arguments:
* `USE_CROSS_ATTN_CONFLICT_LOSS`: whether to use the cross-attention conflict loss
* `OPT`: assign the optimizer for the initial noise optimization, providing `adam` and `rmsprop` options

## Acknowledgments

The code is built upon [InitNO](https://xiefan-guo.github.io/initno), and we adopt the official evaluation prompts from [Attend and Excite](https://github.com/yuval-alaluf/Attend-and-Excite). We thank the authors for open-sourcing.