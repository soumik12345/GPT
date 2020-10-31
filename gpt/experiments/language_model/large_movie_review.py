from glob import glob
from random import shuffle
from ...datasets import TextGenerationDataLoader


class IMDBReviewDataLoader(TextGenerationDataLoader):

    def __init__(self, dataset_url, vocab_size=20000, max_length=100):
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
