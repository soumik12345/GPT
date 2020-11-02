import tensorflow as tf
from ....model import GPT
from .inference import Predictor
from ...pipeline import Experiment
from .callbacks import InferenceCallback
from .dataloader import IMDBReviewDataLoader


class IMDBReviewLanguageExperiment(Experiment):

    def __init__(self):
        super(IMDBReviewLanguageExperiment, self).__init__()
        self.dataset = None
        self.vocabulary = None
        self.word_dictionary = {}
        self.model = None
        self.loss_function = None

    def build_dataset(self, dataset_url, vocab_size=20000, max_length=100):
        loader = IMDBReviewDataLoader(
            dataset_url=dataset_url, vocab_size=vocab_size, max_length=max_length)
        print('Dataset Size: {} files'.format(len(loader)))
        self.dataset, self.vocabulary = loader.get_dataset()
        print('Dataset: {}'.format(self.dataset))
        self._convert_vocab_to_dictionary()

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

    def tokenize(self, start_text):
        start_tokens = [
            self._convert_vocab_to_dictionary().get(_, 1) for _ in start_text.split()]
        return start_tokens

    def train(self, epochs, start_tokens, max_length, max_tokens, top_k, infer_every=1):
        if start_tokens is not None or max_length is not None or\
                max_tokens is not None or top_k is not None or infer_every is not None:
            inference_callback = InferenceCallback(
                start_tokens=start_tokens, max_length=max_length, max_tokens=max_tokens,
                top_k=top_k, infer_every=infer_every, word_dict=self.word_dictionary
            )
        history = self.model.fit(
            self.dataset, epochs=epochs, callbacks=[inference_callback])
        return history

    def _convert_vocab_to_dictionary(self):
        for index, word in enumerate(self.vocabulary):
            self.word_dictionary[word] = index

    def infer(self, start_tokens, max_length, max_tokens, top_k):
        predictor = Predictor(
            max_length=max_length, top_k=top_k,
            word_dict=self.word_dictionary, max_tokens=max_tokens
        )
        prediction = predictor.predict(
            model=self.model, start_tokens=start_tokens
        )
        return prediction
