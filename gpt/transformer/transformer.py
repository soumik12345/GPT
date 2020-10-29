from .attention import tf, MultiHeadAttention


class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, embedding_dimension, num_heads, feed_forward_dimension, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embedding_dimension, num_heads)
        self.feed_forward_network = tf.keras.Sequential([
            tf.keras.layers.Dense(feed_forward_dimension, activation='relu'),
            tf.keras.layers.Dense(embedding_dimension)
        ])
        self.layer_normalization = [
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.LayerNormalization(epsilon=1e-6)
        ]
        self.dropout = [
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Dropout(rate=dropout_rate)
        ]

    def call(self, inputs):
        attention_output = self.dropout[0](self.attention(inputs))
        residual_output = self.layer_normalization[0](inputs + attention_output)
        feed_forward_output = self.dropout[1](self.feed_forward_network(residual_output))
        output = self.layer_normalization[1](residual_output + feed_forward_output)
        return output
