import tensorflow as tf


class CombinedEncoding(tf.keras.layers.Layer):

    def __init__(self, max_length, vocab_size, embedding_dimension):
        super(CombinedEncoding, self).__init__()
        self.text_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dimension)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=max_length, output_dim=embedding_dimension)

    def call(self, inputs):
        max_length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=max_length, delta=1)
        positions = self.position_embedding(positions)
        token = self.text_embedding(inputs)
        output = token + positions
        return output
