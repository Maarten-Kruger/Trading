import os
import re

def clean_filenames():
    folder_path = input("Enter the path to the folder: ").strip()
    
    if not os.path.isdir(folder_path):
        print("Error: That folder does not exist.")
        return

    # Define the pattern to look for at the end of the filename
    # \.       -> a literal dot
    # \d+      -> one or more numbers (e.g., '10')
    # \.       -> a literal dot
    # [A-Fa-f0-9]{30,34} -> 30 to 34 hexadecimal characters (covering the 32 in your example)
    # $        -> ensures this pattern is at the very end of the name
    jibberish_pattern = re.compile(r'\.\d+\.[A-Fa-f0-9]{30,34}$')
    
    # We still define the length to cut, as requested
    CHARS_TO_REMOVE = 36

    count = 0
    skipped = 0
    
    print(f"\nScanning folder: {folder_path}...\n")

    for filename in os.listdir(folder_path):
        old_file_path = os.path.join(folder_path, filename)

        if os.path.isfile(old_file_path):
            name, ext = os.path.splitext(filename)

            # CHECK: Only proceed if the name matches the "jibberish" pattern
            if jibberish_pattern.search(name):
                
                # Double check length just to be safe
                if len(name) > CHARS_TO_REMOVE:
                    new_name_base = name[:-CHARS_TO_REMOVE]
                    new_filename = new_name_base + ext
                    new_file_path = os.path.join(folder_path, new_filename)

                    try:
                        os.rename(old_file_path, new_file_path)
                        print(f"[RENAMED] {filename}")
                        print(f"       -> {new_filename}")
                        count += 1
                    except OSError as e:
                        print(f"Error renaming {filename}: {e}")
            else:
                # If pattern isn't found, we assume it's already clean
                skipped += 1
                # Optional: Uncomment the line below to see skipped files
                # print(f"[SKIPPED] {filename} (Already clean)")

    print(f"\nSummary: {count} files renamed, {skipped} files skipped/clean.")

if __name__ == "__main__":
    clean_filenames()