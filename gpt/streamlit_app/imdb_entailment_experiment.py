import streamlit as st


def imdb_text_entailment_app_module():
    from gpt.experiments.language_model import IMDBReviewLanguageExperiment
    experiment = IMDBReviewLanguageExperiment()
    initialize_wandb = st.checkbox('Initialize Wandb')
    if initialize_wandb:
        project_name = st.text_input('Wandb Project Name', 'gpt')
        experiment_name = st.text_input('Experiemnt Name', 'imdb_language_model')
        wandb_api_key = st.text_input('Wandb API Key', '')
        from gpt.experiments.utils import init_wandb
        st.text('Initializing Wandb...')
        init_wandb(
            project_name=project_name,
            experiment_name=experiment_name,
            wandb_api_key=wandb_api_key
        )
        st.text_input('Done!!!')
    st.text('Fetching and Building IMDB Large Movie Review Dataset...')
    experiment.build_dataset('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    st.text('Done!!!')