import streamlit as st
import os
import pandas as pd
import config
from phase1 import load_and_preprocess_data, find_triggers, generate_images
from phase2 import run_classification_app
from phase3 import run_backtest, generate_simulation_report
from phase4 import run_gallery_app

st.set_page_config(page_title="Trading Strategy Framework", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Phase", ["Phase 1: Data & Images", "Phase 2: Classification", "Phase 3: Backtest", "Phase 4: Gallery View"])

# File uploader in sidebar
st.sidebar.markdown("---")
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV Data File", type=['csv'])

if uploaded_file is not None:
    # Save the uploaded file temporarily to pass path to functions
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state['current_file'] = temp_file_path
    st.session_state['file_name'] = uploaded_file.name
else:
    st.session_state['current_file'] = None

if app_mode == "Phase 1: Data & Images":
    st.title("Phase 1: Data Processing & Image Generation")
    st.write("Upload a CSV file in the sidebar to begin.")

    if st.session_state['current_file']:
        if st.button("Run Phase 1"):
            with st.spinner("Processing data and generating images..."):
                file_path = st.session_state['current_file']
                file_name = st.session_state['file_name']

                df = load_and_preprocess_data(file_path)
                trigger_indices = find_triggers(df)

                progress_bar = st.progress(0)
                progress_text = st.empty()
                progress_text.text("Preparing to generate images...")

                output_dir, count = generate_images(df, trigger_indices, file_name, progress_bar, progress_text)

                # Save output_dir to session state for Phase 2
                st.session_state['output_dir'] = output_dir
                progress_bar.empty()
                progress_text.empty()

                st.success(f"Successfully generated {count} setup images in directory: `{output_dir}`")
    else:
        st.info("Please upload a CSV file.")

elif app_mode == "Phase 2: Classification":
    st.title("Phase 2: Classification App")

    # Find all possible output directories containing a labels.csv
    available_dirs = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and os.path.exists(os.path.join(item, 'labels.csv')):
            available_dirs.append(item)

    if not available_dirs:
        st.warning("No setup folders found. Please run Phase 1 first to generate images.")
    else:
        # Determine index of current output_dir if it exists
        default_idx = 0
        current_dir = st.session_state.get('output_dir')
        if current_dir in available_dirs:
            default_idx = available_dirs.index(current_dir)

        selected_dir = st.selectbox("Select Setup Folder to Classify", available_dirs, index=default_idx)
        st.session_state['output_dir'] = selected_dir

        st.markdown("---")
        run_classification_app(selected_dir)

elif app_mode == "Phase 3: Backtest":
    st.title("Phase 3: Backtesting & Simulation")

    if st.session_state['current_file']:
        if st.button("Run Simulation"):
            with st.spinner("Running backtest simulation..."):
                file_path = st.session_state['current_file']
                file_name = st.session_state['file_name']

                df = load_and_preprocess_data(file_path)
                trigger_indices = find_triggers(df)

                timestamps, equity_curve, drawdowns, trades = run_backtest(df, trigger_indices)

                report_img = generate_simulation_report(timestamps, equity_curve, drawdowns, file_name)

                st.success(f"Simulation completed! Processed {len(trades)} trades.")
                st.image(report_img)
    else:
        st.info("Please upload a CSV file.")

elif app_mode == "Phase 4: Gallery View":
    run_gallery_app()
