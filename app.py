from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from analysis.ai_assistant import generate_ai_answer
from analysis.diagnostics import compare_runs, diagnose_run, supported_columns_message
from analysis.parser import load_runs_from_uploads, load_sample_runs
from analysis.reporting import answer_query, build_comparison_report, build_run_report

st.set_page_config(page_title='TrainScope', page_icon='📈', layout='wide')

st.title('📈 TrainScope')
st.caption('AI-assisted experiment diagnostics for training dynamics, stability, and model comparison.')

with st.sidebar:
    st.header('Controls')
    data_source = st.radio('Choose data source', ['Use sample runs', 'Upload your own CSV files'])
    ranking_metric = st.selectbox('Ranking metric', ['val_accuracy', 'val_loss', 'train_loss'])
    ai_mode = st.toggle('Enable AI analyst', value=True)
    st.markdown('**Expected columns**')
    st.code('epoch, train_loss, val_loss, train_accuracy, val_accuracy')
    st.info('You can upload multiple CSV files to compare different runs.')

runs = []

if data_source == 'Use sample runs':
    runs = load_sample_runs(Path(__file__).parent / 'sample_data')
else:
    uploaded = st.file_uploader(
        'Upload one or more CSV files',
        type=['csv'],
        accept_multiple_files=True,
        help='Each CSV should contain epoch-wise training metrics.',
    )
    if uploaded:
        runs = load_runs_from_uploads(uploaded)

if not runs:
    st.warning('Load sample runs or upload CSV files to begin.')
    st.stop()

st.success(f'Loaded {len(runs)} run(s).')

run_names = [run.name for run in runs]
selected_run_name = st.selectbox('Inspect a run', run_names)
selected_run = next(run for run in runs if run.name == selected_run_name)
selected_diag = diagnose_run(selected_run.data)

col1, col2, col3, col4 = st.columns(4)
col1.metric('Epochs', len(selected_run.data))
col2.metric('Best Val Acc', f"{selected_diag.best_val_accuracy:.4f}" if selected_diag.best_val_accuracy is not None else 'N/A')
col3.metric('Best Val Loss', f"{selected_diag.best_val_loss:.4f}" if selected_diag.best_val_loss is not None else 'N/A')
col4.metric('Stability Score', f"{selected_diag.stability_score:.2f}")

st.subheader('Single Run Diagnostics')
left, right = st.columns([1.4, 1])

with left:
    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    if 'train_loss' in selected_run.data.columns:
        ax_loss.plot(selected_run.data['epoch'], selected_run.data['train_loss'], label='train_loss')
    if 'val_loss' in selected_run.data.columns:
        ax_loss.plot(selected_run.data['epoch'], selected_run.data['val_loss'], label='val_loss')
    ax_loss.set_title(f'Loss Curves · {selected_run.name}')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    st.pyplot(fig_loss)

    fig_acc, ax_acc = plt.subplots(figsize=(8, 4))
    if 'train_accuracy' in selected_run.data.columns:
        ax_acc.plot(selected_run.data['epoch'], selected_run.data['train_accuracy'], label='train_accuracy')
    if 'val_accuracy' in selected_run.data.columns:
        ax_acc.plot(selected_run.data['epoch'], selected_run.data['val_accuracy'], label='val_accuracy')
    ax_acc.set_title(f'Accuracy Curves · {selected_run.name}')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)
    st.pyplot(fig_acc)

with right:
    st.markdown('### Diagnostic Tags')
    for tag in selected_diag.tags:
        st.write(f'- {tag}')

    st.markdown('### What TrainScope sees')
    st.write(build_run_report(selected_run.name, selected_diag))

    if selected_diag.warnings:
        st.markdown('### Warnings')
        for warning in selected_diag.warnings:
            st.warning(warning)

st.divider()
st.subheader('Run Comparison')
comparison_df = compare_runs(runs, ranking_metric)
st.dataframe(comparison_df, use_container_width=True)

st.info(build_comparison_report(comparison_df, ranking_metric))

fig_compare, ax_compare = plt.subplots(figsize=(10, 5))
for run in runs:
    if ranking_metric in run.data.columns:
        ax_compare.plot(run.data['epoch'], run.data[ranking_metric], label=run.name)
ax_compare.set_title(f'Run Comparison · {ranking_metric}')
ax_compare.set_xlabel('Epoch')
ax_compare.set_ylabel(ranking_metric)
ax_compare.legend()
ax_compare.grid(alpha=0.3)
st.pyplot(fig_compare)

st.divider()
st.subheader('Ask TrainScope')
user_query = st.text_input(
    'Ask a question about your runs',
    placeholder='Which run is most stable? Why is one run better? When did overfitting begin?',
)

if user_query:
    if ai_mode:
        response = generate_ai_answer(user_query, runs, comparison_df, ranking_metric)
        st.write(response.answer)
        with st.expander(f'AI evidence · {response.mode}'):
            for item in response.evidence:
                st.write(f'- {item}')
    else:
        st.write(answer_query(user_query, runs, comparison_df, ranking_metric))
else:
    st.caption('Example questions: “Which run is most stable?” “Why is run A better?” “Which run overfit?”')

with st.expander('Debug: supported metrics and file expectations'):
    st.write(supported_columns_message())
    st.dataframe(selected_run.data.head(), use_container_width=True)
