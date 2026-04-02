import streamlit as st
import os
import pandas as pd
import config
from phase1 import load_and_preprocess_data, find_triggers, generate_images
from phase2 import run_classification_app
from phase3 import run_backtest, generate_simulation_report

st.set_page_config(page_title="Trading Strategy Framework", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Phase", ["Phase 1: Data & Images", "Phase 2: Classification", "Phase 3: Backtest"])

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

                output_dir, count = generate_images(df, trigger_indices, file_name)

                # Save output_dir to session state for Phase 2
                st.session_state['output_dir'] = output_dir

                st.success(f"Successfully generated {count} setup images in directory: `{output_dir}`")
    else:
        st.info("Please upload a CSV file.")

elif app_mode == "Phase 2: Classification":
    # If output_dir is in session state, use it. Otherwise try to infer it if a file is uploaded.
    output_dir = st.session_state.get('output_dir')

    if not output_dir and st.session_state['current_file']:
        base_name = os.path.splitext(st.session_state['file_name'])[0]
        inferred_dir = f"{config.STRATEGY_NAME}_{base_name}"
        if os.path.exists(inferred_dir):
            output_dir = inferred_dir
            st.session_state['output_dir'] = output_dir

    run_classification_app(output_dir)

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
