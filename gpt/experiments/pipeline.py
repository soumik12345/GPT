import os
import wandb
import numpy as np
import tensorflow as tf
from ..model import GPT
from wandb.keras import WandbCallback


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
                'Generated Text (Epoch {})'.format(epoch): wandb.Html(prediction)
            })


class LanguageModelExperiment:

    def __init__(self):
        self.dataset = None
        self.vocabulary = None
        self.word_dictionary = {}
        self.model = None
        self.loss_function = None

    def build_dataset(self, dataset_url, vocab_size=20000, max_length=100):
        pass

    def compile(
            self, max_length=100, vocab_size=20000, depth=1, num_heads=2,
            embedding_dimension=256, feed_forward_dimension=256, learning_rate=1e-3):
        self.model = GPT({
            'max_length': max_length,
            'vocab_size': vocab_size,
            'depth': depth, 'num_heads': num_heads,
            'embedding_dimension': embedding_dimension,
            'feed_forward_dimension': feed_forward_dimension
        })
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(
            tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ), [self.loss_function, None]
        )
        self.model.summary()

    def tokenize(self, start_text):
        start_tokens = [
            self.word_dictionary.get(_, 1) for _ in start_text.split()]
        return start_tokens

    def _convert_vocab_to_dictionary(self):
        for index, word in enumerate(self.vocabulary):
            self.word_dictionary[word] = index

    def train(
            self, epochs, start_tokens, max_length,
            max_tokens, top_k, infer_every=1, log_on_wandb=False):
        callbacks = []
        if start_tokens is not None or max_length is not None or \
                max_tokens is not None or top_k is not None or infer_every is not None:
            callbacks.append(
                InferenceCallback(
                    start_tokens=start_tokens, max_length=max_length,
                    max_tokens=max_tokens, top_k=top_k, infer_every=infer_every,
                    word_dict=self.word_dictionary, log_on_wandb=log_on_wandb
                )
            )
        if log_on_wandb:
            checkpoint_path = os.path.join(wandb.run.dir, 'gpt_language_model_checkpoint') + '{epoch}'
            callbacks.append(WandbCallback())
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path, save_weights_only=True,
                    save_best_only=False, monitor='loss', verbose=1
                )
            )
        history = self.model.fit(
            self.dataset, epochs=epochs, callbacks=callbacks)
        return history

    def infer(self, start_tokens, max_length, max_tokens, top_k):
        predictor = Predictor(
            max_length=max_length, top_k=top_k,
            word_dict=self.word_dictionary, max_tokens=max_tokens
        )
        prediction = predictor.predict(
            model=self.model, start_tokens=start_tokens
        )
        return prediction
