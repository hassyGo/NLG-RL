# NLG-RL
This repository provides PyTorch implementations of our method [1] for reinforcement learning for sentence generation with action-space reduction.

## Requirements
* Python 3.X
* PyTorch 0.2X or 0.4X
* Numba

## Training NMT models

Please `cd` to `./code_0.2` or `./code_0.4` to run experiments.
The input data format is <b>one sentence per line</b> for each language.

### Small softmax
* Vocabulary predictor (minimal usage with the default settings used in our paper)<br>
`python train_vocgen.py --train_source XX --train_target YY --dev_source ZZ --dev_target WW`<br><br>
* NMT with cross-entropy (minimal usage <b>after training the vocabulary predictor</b>)<br>
`python train_nmt.py --train_source XX --train_target YY --dev_source ZZ --dev_target WW`<br><br>
* NMT with REINFORCE and cross-entropy<br>
`python train_nmt_rl.py --train_source XX --train_target YY --dev_source ZZ --dev_target WW`<br><br>

### Full softmax (standard baseline)
* NMT with cross-entropy (minimal usage <b>without using the vocabulary predictor</b>)<br>
`python train_nmt.py --K -1 --train_source XX --train_target YY --dev_source ZZ --dev_target WW`<br><br>
* NMT with REINFORCE and cross-entropy<br>
`python train_nmt_rl.py --K -1 --train_source XX --train_target YY --dev_source ZZ --dev_target WW`<br><br>

## Reference
[1] <b>Kazuma Hashimoto</b> and Yoshimasa Tsuruoka. 2019. Accelerated Reinforcement Learning for Sentence Generation by Vocabulary Prediction. In <i>Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (<b>NAACL-HLT 2019</b>)</i>, <a href="https://arxiv.org/abs/1809.01694">arXiv cs.CL 1809.01694<a/>. (<a href="http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2018fastrl/bibtex.bib">bibtex</a>)

## Questions?
Any issues and PRs are welcome.

E-mail: hassy@logos.t.u-tokyo.ac.jp
