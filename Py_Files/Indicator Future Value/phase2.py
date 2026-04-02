import streamlit as st
import pandas as pd
import os

def load_labels(output_dir):
    labels_file = os.path.join(output_dir, 'labels.csv')
    if os.path.exists(labels_file):
        df = pd.read_csv(labels_file)
        return df
    return pd.DataFrame()

def save_labels(output_dir, df):
    labels_file = os.path.join(output_dir, 'labels.csv')
    df.to_csv(labels_file, index=False)

def run_classification_app(output_dir):
    if not output_dir or not os.path.exists(output_dir):
        st.warning("Please run Phase 1 first to generate images.")
        return

    df_labels = load_labels(output_dir)

    if df_labels.empty:
        st.warning("No labels found in the selected folder.")
        return

    # State management for current image index
    if 'current_img_idx' not in st.session_state:
        st.session_state.current_img_idx = 0

    if 'custom_bins' not in st.session_state:
        st.session_state.custom_bins = []

    # 3-Column Layout
    col_left, col_center, col_right = st.columns([1, 2, 1])

    # --- LEFT COLUMN: File Explorer ---
    with col_left:
        st.subheader("File Explorer")

        # Filter buttons
        filter_status = st.radio("Classification Filter", ["All", "Unclassified", "Classified"])

        # Signal filter
        if 'Signal' in df_labels.columns:
            signal_options = ["All"] + sorted(df_labels['Signal'].dropna().unique().tolist())
            signal_filter = st.radio("Signal Filter", signal_options)
        else:
            signal_filter = "All"

        if filter_status == "Unclassified":
            display_df = df_labels[df_labels['Label'] == 'Unclassified']
        elif filter_status == "Classified":
            classified_df = df_labels[df_labels['Label'] != 'Unclassified']

            # Secondary filter for specific bins
            if not classified_df.empty:
                unique_labels = sorted(classified_df['Label'].unique().tolist())
                selected_bin_filter = st.selectbox("Filter by Bin", ["All Classified"] + unique_labels)

                if selected_bin_filter != "All Classified":
                    display_df = classified_df[classified_df['Label'] == selected_bin_filter]
                else:
                    display_df = classified_df
            else:
                display_df = classified_df
        else:
            display_df = df_labels

        if signal_filter != "All" and 'Signal' in display_df.columns:
            display_df = display_df[display_df['Signal'] == signal_filter]

        st.write(f"Showing {len(display_df)} setups")

        # Order toggle
        order_mode = st.radio("Order", ["Sequential", "Random"])

        # Create a list of options for the selectbox
        options = []
        for idx, row in display_df.iterrows():
            mark = "❌" if row['Label'] == 'Unclassified' else "✅"
            options.append(f"{idx}: {mark} {row['Image_Name']}")

        if options:
            if order_mode == "Random":
                import random
                # Use a stable seed for random choice based on session state if possible
                if 'random_options' not in st.session_state or st.session_state.get('last_filter') != filter_status or st.session_state.get('last_order') != order_mode:
                    shuffled = list(options)
                    random.shuffle(shuffled)
                    st.session_state.random_options = shuffled

                display_options = st.session_state.random_options
                # Filter display options to only those still valid
                display_options = [opt for opt in display_options if opt in options]
                st.session_state.random_options = display_options
            else:
                display_options = options

            st.session_state.last_filter = filter_status
            st.session_state.last_order = order_mode

            # Find the index of the currently selected image in the options
            current_option_idx = 0
            for i, opt in enumerate(display_options):
                if int(opt.split(":")[0]) == st.session_state.current_img_idx:
                    current_option_idx = i
                    break

            selected_option = st.selectbox("Select Setup", display_options, index=current_option_idx)
            # Extract the actual index from the string (e.g., "5: ❌ sample_006.png" -> 5)
            selected_idx = int(selected_option.split(":")[0])
            st.session_state.current_img_idx = selected_idx

            # Update the list index for navigation
            current_list_idx = display_options.index(selected_option)
        else:
            st.info("No setups matching filter.")

    # --- CENTER COLUMN: Image Viewer & Classification ---
    with col_center:
        if not display_df.empty:
            idx = st.session_state.current_img_idx
            current_row = df_labels.iloc[idx]
            image_name = current_row['Image_Name']
            image_path = os.path.join(output_dir, image_name)

            st.subheader(f"Reviewing: {image_name}")
            st.write(f"Time: {current_row['Time']} | Current Label: **{current_row['Label']}**")

            if os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
            else:
                st.error(f"Image not found: {image_path}")

            st.write("### Classify")

            # Default Bins
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("Up", use_container_width=True):
                    df_labels.at[idx, 'Label'] = "Up"
                    save_labels(output_dir, df_labels)
                    st.rerun()
            with btn_col2:
                if st.button("Down", use_container_width=True):
                    df_labels.at[idx, 'Label'] = "Down"
                    save_labels(output_dir, df_labels)
                    st.rerun()
            with btn_col3:
                if st.button("Sideways", use_container_width=True):
                    df_labels.at[idx, 'Label'] = "Sideways"
                    save_labels(output_dir, df_labels)
                    st.rerun()

            # Custom Bins
            if st.session_state.custom_bins:
                st.write("Custom Bins")
                custom_cols = st.columns(min(3, len(st.session_state.custom_bins)))
                for i, custom_bin in enumerate(st.session_state.custom_bins):
                    col_idx = i % 3
                    with custom_cols[col_idx]:
                        if st.button(custom_bin, key=f"btn_{custom_bin}", use_container_width=True):
                            df_labels.at[idx, 'Label'] = custom_bin
                            save_labels(output_dir, df_labels)
                            st.rerun()

            # Add new custom bin
            new_bin = st.text_input("Add Custom Bin")
            if st.button("Add Bin"):
                if new_bin and new_bin not in st.session_state.custom_bins:
                    st.session_state.custom_bins.append(new_bin)
                    st.rerun()

            # Navigation
            if options:
                nav1, nav2, nav3 = st.columns(3)
                with nav1:
                    if st.button("Previous"):
                        prev_list_idx = max(0, current_list_idx - 1)
                        st.session_state.current_img_idx = int(display_options[prev_list_idx].split(":")[0])
                        st.rerun()
                with nav3:
                    if st.button("Next"):
                        next_list_idx = min(len(display_options) - 1, current_list_idx + 1)
                        st.session_state.current_img_idx = int(display_options[next_list_idx].split(":")[0])
                        st.rerun()

    # --- RIGHT COLUMN: Statistics ---
    with col_right:
        st.subheader("Statistics")
        total = len(df_labels)
        unclassified = len(df_labels[df_labels['Label'] == 'Unclassified'])
        classified = total - unclassified

        st.write(f"**Total Setups:** {total}")
        st.write(f"**Classified:** {classified}")
        st.write(f"**Unclassified:** {unclassified}")

        st.progress(classified / total if total > 0 else 0)

        st.write("---")
        st.write("**Counts by Label:**")
        label_counts = df_labels['Label'].value_counts()
        for label, count in label_counts.items():
            st.write(f"- {label}: {count}")
