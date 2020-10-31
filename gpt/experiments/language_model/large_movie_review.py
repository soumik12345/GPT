from glob import glob
from random import shuffle
from ...model import GPT, tf
from ..pipeline import Experiment
from ...datasets import TextGenerationDataLoader


class IMDBReviewDataLoader(TextGenerationDataLoader):

    def __init__(self, dataset_url, vocab_size, max_length):
        super(IMDBReviewDataLoader, self).__init__(
            dataset_url=dataset_url, vocab_size=vocab_size, max_length=max_length)

    def get_text_files(self):
        dataset_path = '/'.join(self.dataset_path.split('/')[:-1]) + '/aclImdb'
        text_files = [
            *glob(dataset_path + '/train/pos/*.txt'),
            *glob(dataset_path + '/train/neg/*.txt'),
            *glob(dataset_path + '/test/pos/*.txt'),
            *glob(dataset_path + '/test/neg/*.txt')
        ]
        shuffle(text_files)
        return text_files


class IMDBReviewLanguageModel(Experiment):

    def __init__(self):
        super(IMDBReviewDataLoader, self).__init__()
        self.dataset = None
        self.vocabulary = None
        self.model = None
        self.loss_function = None

    def build_dataset(self, dataset_url, vocab_size=20000, max_length=100):
        loader = IMDBReviewDataLoader(
            dataset_url=dataset_url, vocab_size=vocab_size, max_length=max_length)
        print('Dataset Size: {} files'.format(len(loader)))
        self.dataset, self.vocabulary = loader.get_dataset()
        print('Dataset: {}'.format(self.dataset))

    def build_model(
            self, max_length=100, vocab_size=20000, depth=1, num_heads=2,
            embedding_dimension=256, feed_forward_dimension=256):
        self.model = GPT({
            'max_length': max_length,
            'vocab_size': vocab_size,
            'depth': depth, 'num_heads': num_heads,
            'embedding_dimension': embedding_dimension,
            'feed_forward_dimension': feed_forward_dimension
        })

    def compile(self, learning_rate=1e-3):
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(
            tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ), [self.loss_function, None]
        )
