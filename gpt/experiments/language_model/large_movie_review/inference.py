import numpy as np
import tensorflow as tf


class Predictor:

    def __init__(self, max_length, max_tokens, top_k, word_dict):
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.word_dict = word_dict

    def sample_tokens_from_logits(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.top_k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        predictions = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        predictions = np.asarray(predictions).astype("float32")
        samples = np.random.choice(indices, p=predictions)
        return samples

    def decode(self, token):
        token_dict = {v: k for k, v in self.word_dict.items()}
        return token_dict[token]

    def predict(self, model, start_tokens):
        start_tokens = [_ for _ in start_tokens]
        num_tokens_generated, tokens_generated = 0, []
        while num_tokens_generated <= self.max_tokens:
            pad_len = self.max_length - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.max_length]
                sample_index = self.max_length - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = model.predict(x)
            sample_token = self.sample_tokens_from_logits(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        predicted_text = " ".join(
            [self.decode(_) for _ in start_tokens + tokens_generated]
        )
        return predicted_text
