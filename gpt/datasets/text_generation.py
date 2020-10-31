from glob import glob
import tensorflow as tf
from .utils import standardize
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


class TextGenerationDataLoader:

    def __init__(self, dataset_url, vocab_size, max_length):
        self.dataset_url = dataset_url
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dataset_path = self.download_dataset()
        self.text_files = self.get_text_files()
        self.vecotization_layer = None

    def download_dataset(self):
        dataset_path = tf.keras.utils.get_file(
            self.dataset_url.split('/')[-1], self.dataset_url, extract=True)
        return dataset_path

    def get_text_files(self):
        return glob(self.dataset_url + '/*.txt')

    def __len__(self):
        return len(self.text_files)

    def build_vecotization_layer(self):
        vectorization_layer = TextVectorization(
            standardize=standardize,
            max_tokens=self.vocab_size - 1,
            output_mode="int",
            output_sequence_length=self.max_length + 1,
        )
        return vectorization_layer

    def map_function(self, text):
        text = tf.expand_dims(text, -1)
        tokenized_sentences = self.vecotization_layer(text)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y

    def get_dataset(self, buffer_size=256, batch_size=32):
        dataset = tf.data.TextLineDataset(self.text_files)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size=batch_size)
        self.vecotization_layer = self.build_vecotization_layer()
        self.vecotization_layer.adapt(dataset)
        vocabulary = self.vecotization_layer.get_vocabulary()
        dataset = dataset.map(self.map_function)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, vocabulary
