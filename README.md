# GPT (Ongoing)

Tensorflow implementation of Generative Pre-Training on GPT.

# Experiments

## Language Model

```python
from gpt.experiments.language_model import IMDBReviewLanguageExperiment

experiment = IMDBReviewLanguageExperiment()
experiment.build_dataset('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
experiment.compile()
start_text = 'the actor was'
start_tokens = experiment.tokenize(start_text=start_text)
experiment.train(
    epochs=30, start_tokens=start_tokens,
    max_length=100, max_tokens=40, top_k=10, infer_every=1
)
```