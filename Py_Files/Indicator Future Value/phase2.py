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

    # Extract all unique bins currently in the dataset to populate custom_bins
    existing_bins = set()
    for label_str in df_labels['Label'].dropna():
        if label_str != 'Unclassified':
            # Labels could be comma-separated
            bins = [b.strip() for b in label_str.split(',') if b.strip()]
            existing_bins.update(bins)

    # Remove default bins from custom bins list
    default_bins = ["Up", "Down", "Sideways"]
    custom_bins_from_file = [b for b in existing_bins if b not in default_bins]

    if 'custom_bins' not in st.session_state:
        st.session_state.custom_bins = custom_bins_from_file
    else:
        # Merge file bins with session state bins
        for b in custom_bins_from_file:
            if b not in st.session_state.custom_bins:
                st.session_state.custom_bins.append(b)

    all_available_bins = default_bins + st.session_state.custom_bins

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

            # Secondary filter for specific bins (Checkboxes with AND logic)
            if not classified_df.empty:
                st.write("Filter by Bins (AND logic):")
                selected_filter_bins = []

                # Checkboxes for bin filtering
                filter_cols = st.columns(2)
                for i, bin_name in enumerate(all_available_bins):
                    with filter_cols[i % 2]:
                        if st.checkbox(bin_name, key=f"filter_{bin_name}"):
                            selected_filter_bins.append(bin_name)

                if selected_filter_bins:
                    # Filter logic: Setup must contain ALL selected bins
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

        if signal_filter != "All" and 'Signal' in display_df.columns:
            display_df = display_df[display_df['Signal'] == signal_filter]

        st.write(f"Showing {len(display_df)} setups")

        # Order toggle
        order_mode = st.radio("Order", ["Sequential", "Random"])

        # Create a list of options for the selectbox
        # Get all valid indices currently in the filtered view
        valid_indices = display_df.index.tolist()

        if valid_indices:
            if order_mode == "Random":
                import random
                # Reshuffle if filter or order mode changes
                if 'random_indices' not in st.session_state or st.session_state.get('last_filter') != filter_status or st.session_state.get('last_order') != order_mode:
                    shuffled = list(valid_indices)
                    random.shuffle(shuffled)
                    st.session_state.random_indices = shuffled

                # Keep only valid indices, preserving random order
                display_indices = [i for i in st.session_state.random_indices if i in valid_indices]
                
                # Add any new indices that might have appeared (just in case)
                for i in valid_indices:
                    if i not in display_indices:
                        display_indices.append(i)
                        
                st.session_state.random_indices = display_indices
            else:
                display_indices = valid_indices

            st.session_state.last_filter = filter_status
            st.session_state.last_order = order_mode

            # Build display_options using the ordered indices
            display_options = []
            for i in display_indices:
                row = display_df.loc[i]
                mark = "❌" if row['Label'] == 'Unclassified' else "✅"
                display_options.append(f"{i}: {mark} {row['Image_Name']}")
                
            options = display_options # Keep references intact for the code below

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

            # Determine currently checked bins for this image
            current_label = df_labels.at[idx, 'Label']
            if current_label == 'Unclassified' or pd.isna(current_label):
                current_checked = []
            else:
                current_checked = [b.strip() for b in current_label.split(',')]

            new_checked = []

            # Default Bins
            st.write("Default Bins")
            cb_cols = st.columns(3)
            for i, bin_name in enumerate(default_bins):
                with cb_cols[i]:
                    is_checked = st.checkbox(bin_name, value=(bin_name in current_checked), key=f"cb_{bin_name}_{idx}")
                    if is_checked:
                        new_checked.append(bin_name)

            # Custom Bins
            if st.session_state.custom_bins:
                st.write("Custom Bins")
                custom_cols = st.columns(3)
                for i, custom_bin in enumerate(st.session_state.custom_bins):
                    with custom_cols[i % 3]:
                        is_checked = st.checkbox(custom_bin, value=(custom_bin in current_checked), key=f"cb_{custom_bin}_{idx}")
                        if is_checked:
                            new_checked.append(custom_bin)

            # Check if selection changed
            if set(new_checked) != set(current_checked):
                if not new_checked:
                    df_labels.at[idx, 'Label'] = 'Unclassified'
                else:
                    df_labels.at[idx, 'Label'] = ', '.join(new_checked)
                save_labels(output_dir, df_labels)
                st.rerun()

            # Add new custom bin
            new_bin = st.text_input("Add Custom Bin")
            if st.button("Add Bin"):
                if new_bin and new_bin not in st.session_state.custom_bins and new_bin not in default_bins:
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
        st.subheader("Statistics (Filtered)")

        # Calculate statistics based on the FILTERED view (display_df)
        total = len(display_df)
        unclassified = len(display_df[display_df['Label'] == 'Unclassified'])
        classified = total - unclassified

        st.write(f"**Total Filtered:** {total}")
        st.write(f"**Classified:** {classified}")
        st.write(f"**Unclassified:** {unclassified}")

        st.progress(classified / total if total > 0 else 0)

        st.write("---")
        st.write("**Counts by Bin:**")

        # Count individual bins from comma-separated strings
        bin_counts = {}
        for label_str in display_df['Label'].dropna():
            if label_str == 'Unclassified':
                continue
            bins = [b.strip() for b in label_str.split(',')]
            for b in bins:
                bin_counts[b] = bin_counts.get(b, 0) + 1

        # Sort by count descending
        sorted_counts = sorted(bin_counts.items(), key=lambda item: item[1], reverse=True)
        for label, count in sorted_counts:
            st.write(f"- {label}: {count}")

        if not sorted_counts and classified == 0:
            st.write("No classified labels in this view.")
