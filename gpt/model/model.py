from .transformer import TransformerBlock
from .embedding import tf, CombinedEncoding


def GPT(model_configs):

    input_tensor = tf.keras.Input(
        shape=(model_configs['max_length']),
        dtype=tf.int32, name='GPT_Input'
    )

    x = CombinedEncoding(
        max_length=model_configs['max_length'],
        vocab_size=model_configs['vocab_size'],
        embedding_dimension=model_configs['embedding_dimension']
    )(input_tensor)

    for _ in range(model_configs['depth']):
        x = TransformerBlock(
            embedding_dimension=model_configs['embedding_dimension'],
            num_heads=model_configs['num_heads'],
            feed_forward_dimension=model_configs['feed_forward_dimension']
        )(x)

    output_tensor = tf.keras.layers.Dense(
        model_configs['vocab_size'], name='GPT_Output'
    )(x)

    model = tf.keras.Model(
        input_tensor, [output_tensor, x],
        name='GPT_depth_{}'.format(model_configs['depth'])
    )
    return model
