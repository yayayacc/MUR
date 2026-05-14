<h1 align="center">
MUR: Momentum Uncertainty Guided Reasoning for Large Language Models
</h1>

<p align="center">
  <!-- <a href="https://github.com/yayayacc/MUR/"><b>[🌐 PyPi Package]</b></a> • -->
  <a href="https://arxiv.org/abs/2507.14958"><b>[📜 Paper]</b></a> •
  <a href="https://github.com/yayayacc/MUR/"><b>[🐱 GitHub]</b></a>
  
</p>

<p align="center"> Repo for "MUR: Momentum Uncertainty Guided Reasoning for Large Language Models</a>"</p>
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

Feel free to contact with me if you have any questions ~~~

