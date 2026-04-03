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

    # Filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)

    with col1:
        filter_status = st.radio("Classification Filter", ["All", "Unclassified", "Classified"])

    with col2:
        if 'Signal' in df_labels.columns:
            signal_options = ["All"] + sorted(df_labels['Signal'].dropna().unique().tolist())
            signal_filter = st.radio("Signal Filter", signal_options)
        else:
            signal_filter = "All"

    # Extract all unique bins for checkbox filtering
    existing_bins = set()
    for label_str in df_labels['Label'].dropna():
        if label_str != 'Unclassified':
            bins = [b.strip() for b in label_str.split(',') if b.strip()]
            existing_bins.update(bins)

    all_available_bins = sorted(list(existing_bins))
    selected_filter_bins = []

    if filter_status == "Unclassified":
        display_df = df_labels[df_labels['Label'] == 'Unclassified']
    elif filter_status == "Classified":
        classified_df = df_labels[df_labels['Label'] != 'Unclassified']

        if not classified_df.empty:
            st.write("Filter by Bins (AND logic):")
            # Checkboxes for bin filtering
            filter_cols = st.columns(min(len(all_available_bins), 6)) if all_available_bins else []
            for i, bin_name in enumerate(all_available_bins):
                with filter_cols[i % len(filter_cols)]:
                    if st.checkbox(bin_name, key=f"gal_filter_{bin_name}"):
                        selected_filter_bins.append(bin_name)

            if selected_filter_bins:
                def contains_all_bins(label_str):
                    if label_str == 'Unclassified': return False
                    setup_bins = [b.strip() for b in label_str.split(',')]
                    return all(b in setup_bins for b in selected_filter_bins)

                mask = classified_df['Label'].apply(contains_all_bins)
                display_df = classified_df[mask]
            else:
                display_df = classified_df
        else:
            display_df = classified_df
    else:
        display_df = df_labels

    # Apply signal filter
    if signal_filter != "All" and 'Signal' in display_df.columns:
        display_df = display_df[display_df['Signal'] == signal_filter]

    st.markdown("---")

    filter_desc = filter_status
    if signal_filter != "All":
        filter_desc += f" | Signal: {signal_filter}"
    if selected_filter_bins:
        filter_desc += f" | Bins: {', '.join(selected_filter_bins)}"

    st.write(f"### Gallery: {filter_desc} ({len(display_df)} images)")

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
                    label_text = f"\nLabels: {row['Label']}" if row['Label'] != 'Unclassified' else "\nUnclassified"
                    st.image(image_path, caption=f"{image_name} | {row['Time']}{signal_text}{label_text}", use_container_width=True)
                else:
                    st.error(f"Image not found: {image_name}")
    else:
        st.info("No images match the selected filters.")
