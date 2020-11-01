# GPT (Ongoing)

Tensorflow implementation of Generative Pre-Training on GPT.

# Experiments

## Language Model

```python
from gpt.experiments.language_model import IMDBReviewLanguageExperiment

experiment = IMDBReviewLanguageExperiment()
experiment.build_dataset('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
experiment.compile()
```