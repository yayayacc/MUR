<h1 align="center">
MUR: Momentum Uncertainty Guided Reasoning For Large
Language Models
</h1>

<p align="center">
  <!-- <a href="https://github.com/yayayacc/MUR/"><b>[🌐 PyPi Package]</b></a> • -->
  <a href="https://arxiv.org/abs/2507.14958"><b>[📜 Paper]</b></a> •
  <a href="https://github.com/yayayacc/MUR/"><b>[🐱 GitHub]</b></a>
  
</p>

<p align="center"> Repo for "MUR: Momentum Uncertainty Guided Reasoning For Large Language Models</a>"</p>
<a href="https://arxiv.org/abs/2507.14958" target="_blank">

## 🔥 News

- [2025/07/22] 🔥🔥🔥 Our paper is released !!!
- [2025/07/19] 🔥🔥🔥 Our github repo is released!!!

## 📖 Results

MUR reduces computation by over 50\% on average across three backbone models, while improving accuracy by 0.62–3.37\%.

<p align="center">
    <img src="./assets/Intro.png" alt="scaling" width="400">
</p>

## 🚀 Quick Start

To use MUR, we can try with the following command.

Firstly, create the environment and install the requirements. This implementation is accelerated and supported by vllm.

```bash
# env
conda create -n mur python==3.11.9
conda activate mur
pip install -r requirements.txt
```

Next, simply run different python files: 

```python
python [TTS setting]-[per_step_scale|mur].py
```

Finally, run eval files. To be specific, please eval gpqa_diamond dataset using ``eval/eval_gpqa_cot.py``. Adiitionaly, use ``eval/math_verifier.py`` to verify math datasets.

We are resturing our code to make it more readable, so it may have some trouble in main branch(which we are restrctruing). If you find sth wrong in the main branch, you can refer to the code in the master branch(which is the original code). Thanks!

Feel free to contact with me if you have any questions ~~~

## Citation

If you find it helpful, please kindly cite our paper.
```
@article{yan2025mur,
  title={MUR: Momentum Uncertainty guided Reasoning for Large Language Models},
  author={Hang Yan, Fangzhi Xu, Rongman Xu, Yifei Li, Jian Zhang, Haoran Luo, Xiaobao Wu, Luu Anh Tuan, Haiteng Zhao, Qika Lin, Jun Liu},
  journal={arXiv preprint arXiv:2507.14958},
  year={2025}
}
```
