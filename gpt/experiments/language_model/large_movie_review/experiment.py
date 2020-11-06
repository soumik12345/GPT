from .dataloader import IMDBReviewDataLoader
from ...pipeline import LanguageModelExperiment


class IMDBReviewLanguageExperiment(LanguageModelExperiment):

    def __init__(self):
        super(IMDBReviewLanguageExperiment, self).__init__()

    def build_dataset(self, dataset_url, vocab_size=20000, max_length=100):
        loader = IMDBReviewDataLoader(
            dataset_url=dataset_url, vocab_size=vocab_size, max_length=max_length)
        print('Dataset Size: {} files'.format(len(loader)))
        self.dataset, self.vocabulary = loader.get_dataset()
        print('Dataset: {}'.format(self.dataset))
        self._convert_vocab_to_dictionary()
