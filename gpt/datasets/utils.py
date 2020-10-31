from string import punctuation
from .text_generation import tf


def standardize(input_string):
    lowercase_string = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercase_string, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{punctuation}])", r" \1")


