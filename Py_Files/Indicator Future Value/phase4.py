import streamlit as st
import pandas as pd
import os
from phase2 import load_labels

def run_gallery_app():
    st.title("Phase 4: Gallery View")

    # Find all possible output directories containing a labels.csv
    available_dirs = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and os.path.exists(os.path.join(item, 'labels.csv')):
            available_dirs.append(item)

    if not available_dirs:
        st.warning("No setup folders found. Please run Phase 1 first to generate images.")
        return

    # Select Directory
    default_idx = 0
    current_dir = st.session_state.get('output_dir')
    if current_dir in available_dirs:
        default_idx = available_dirs.index(current_dir)

    selected_dir = st.selectbox("Select Setup Folder", available_dirs, index=default_idx)
    st.session_state['output_dir'] = selected_dir

    df_labels = load_labels(selected_dir)
    if df_labels.empty:
        st.warning("No labels found in the selected folder.")
        return

    st.markdown("---")

    # Select Bin / Filter
    classified_df = df_labels[df_labels['Label'] != 'Unclassified']

    if classified_df.empty:
        st.info("No classified images in this folder. Please classify images in Phase 2 first.")
        return

    unique_labels = sorted(classified_df['Label'].unique().tolist())
    selected_bin = st.selectbox("Select Classification Bin to View", unique_labels)

    display_df = classified_df[classified_df['Label'] == selected_bin]

    st.write(f"### Gallery: '{selected_bin}' ({len(display_df)} images)")

    # Display images in 3 columns
    if not display_df.empty:
        cols = st.columns(3)
        for index, (_, row) in enumerate(display_df.iterrows()):
            image_name = row['Image_Name']
            image_path = os.path.join(selected_dir, image_name)

            col = cols[index % 3]

            with col:
                if os.path.exists(image_path):
                    signal_text = f" | Signal: {row['Signal']}" if 'Signal' in row and pd.notna(row['Signal']) and row['Signal'] != "Unknown" else ""
                    st.image(image_path, caption=f"{image_name} | {row['Time']}{signal_text}", use_container_width=True)
                else:
                    st.error(f"Image not found: {image_name}")
    else:
        st.info(f"No images found for bin '{selected_bin}'.")
