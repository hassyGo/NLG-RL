# NLG-RL
This repository provides PyTorch implementations of our method [1] for reinforcement learning for sentence generation with action-space reduction.

## Requirements
* Python 3.X
* PyTorch 0.2X or 0.4X
* Numba

## Training NMT models
* Vocabulary predictor (minimal usage with the default settings used in our paper)<br>
`python train_vocgen.py --train_source XX --train_target YY --dev_source ZZ --dev_target WW`

---- Under construction. ----

## Reference
[1] <b>Kazuma Hashimoto</b> and Yoshimasa Tsuruoka. 2018. Accelerated Reinforcement Learning for Sentence Generation by Vocabulary Prediction. <a href="https://arxiv.org/abs/1809.01694">arXiv cs.CL 1809.01694<a/>. (<a href="http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2018fastrl/bibtex.bib">bibtex</a>)

## Questions?
Any issues and PRs are welcome.

E-mail: hassy@logos.t.u-tokyo.ac.jp
