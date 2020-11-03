import streamlit as st


def demo_app():
    st.markdown('# GPT')
    st.markdown('<hr />', unsafe_allow_html=True)
    option = st.selectbox(
        'Select your action:', (
            '', 'Train a Text Entailment Model',
            'Infer from a Text Entailment Model'
        )
    )
    if option == 'Train a Text Entailment Model':
        experiment_option = st.selectbox(
            'Select Experiment:', (
                '', 'IMDB Large Movie Review'
            )
        )
        if experiment_option == 'IMDB Large Movie Review':
            from gpt.streamlit_app import imdb_text_entailment_app_module
            imdb_text_entailment_app_module()


demo_app()
