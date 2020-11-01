import tensorflow as tf
from .inference import Predictor


class InferenceCallback(tf.keras.callbacks.Callback):

    def __init__(self, start_tokens, max_length, max_tokens, top_k, word_dict, infer_every=1):
        super(InferenceCallback, self).__init__()
        self.start_tokens = start_tokens
        self.infer_every = infer_every
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
        print('Sample Generate Text:', prediction)
