import tensorflow as tf


def causal_attention_mask(destination_dimension, source_dimension, output_type):
    i = tf.range(destination_dimension)[:, None]
    j = tf.range(source_dimension)
    mask = i >= j - source_dimension + destination_dimension
    mask = tf.cast(mask, output_type)
    return mask


def attention(query, key, value):
    # Calculate Scaled Score
    score = tf.matmul(query, key, transpose_b=True) # Q * Transpose(K)
    key_dimension = tf.cast(tf.shape(key)[-1], tf.float32) # d_k
    scaled_score = score / tf.math.sqrt(key_dimension) # Q * Transpose(K) / (d_K ^ 0.5)

    # Apply Causal Attention Mask to prevent information flow from future tokens
    score_shape = tf.shape(scaled_score)
    attention_mask = causal_attention_mask(score_shape[2], score_shape[3], scaled_score.dtype)
    attention_mask = tf.reshape(attention_mask, [1, 1, score_shape[2], score_shape[3]])
    # Apply Mask to scaled score
    scaled_score = scaled_score * attention_mask - 1e4 * (1 - attention_mask)

    weights = tf.nn.softmax(scaled_score, axis=-1)
    output = tf.matmul(weights, value)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, embedding_dimension, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert embedding_dimension % num_heads == 0, 'embedding_dimension should be divisible by num_heads'
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.projection_dimension = embedding_dimension // num_heads

        self.query_layer = tf.keras.layers.Dense(embedding_dimension)
        self.key_layer = tf.keras.layers.Dense(embedding_dimension)
        self.value_layer = tf.keras.layers.Dense(embedding_dimension)

        self.output_layer = tf.keras.layers.Dense(embedding_dimension)

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (
            batch_size, -1, self.num_heads, self.projection_dimension))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # (batch_size, sequence_length, embedding_dimension)
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)

        # (batch_size, num_heads, sequence_length, embedding_dimension)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention_output, _ = attention(query, key, value)
        # (batch_size, sequence_length, num_heads, projection_dimension)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        # Join heads (batch_size, sequence_length, projection_dimension)
        attention_output = tf.reshape(
            attention_output, (batch_size, -1, self.embedding_dimension))

        output = self.output_layer(attention_output)
        return output
