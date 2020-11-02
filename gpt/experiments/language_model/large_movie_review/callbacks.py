import wandb
import tensorflow as tf
from .inference import Predictor


class InferenceCallback(tf.keras.callbacks.Callback):

    def __init__(
            self, start_tokens, max_length, max_tokens,
            top_k, word_dict, infer_every=1, log_on_wandb=False):
        super(InferenceCallback, self).__init__()
        self.start_tokens = start_tokens
        self.infer_every = infer_every
        self.log_on_wandb = log_on_wandb
        self.predictor = Predictor(
            max_length=max_length, top_k=top_k,
            word_dict=word_dict, max_tokens=max_tokens
        )

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.infer_every != 0:
            return
        prediction = self.predictor.predict(
            model=self.model, start_tokens=self.start_tokens
        )
        print('\nSample Generate Text:', prediction)
        if self.log_on_wandb:
            wandb.log({
                "custom_string": wandb.Html(
                    'Generated Text: {}'.format(prediction)
                )
            })
