# --- CONFIGURATION ---
# 1. Paste your path between the quotes below. 
# 2. Keep the 'r' before the quotes! It fixes the crash you just had.
file_path = r"C:\Users\Maarten\AppData\Roaming\MetaQuotes\Terminal\53785E099C927DB68A545C249CDBCE06\MQL5\Experts\Trading\Trading_Simulator\EURUSD_H1_200.csv"
# ---------------------

def convert_csv():
    try:
        with open(file_path, 'r') as file:
            print("--- COPY BELOW THIS LINE ---")
            
            for line in file:
                # Remove invisible line breaks at the end
                clean_line = line.strip()
                
                # Skip empty lines
                if not clean_line:
                    continue

                # Split the data by the semicolon separator
                parts = clean_line.split(';')

                # Check if we have all 5 columns (Time, Open, High, Low, Close)
                if len(parts) >= 5:
                    time = parts[0]
                    # Replace European commas with dots for the numbers
                    open_price = parts[1].replace(',', '.')
                    high = parts[2].replace(',', '.')
                    low = parts[3].replace(',', '.')
                    close = parts[4].replace(',', '.')

                    # Print in the clean format: Time, Open, High, Low, Close
                    print(f"{time}; {open_price}; {high}; {low}; {close}")

            print("--- END OF FILE ---")

    except FileNotFoundError:
        print(f"Error: Could not find file at: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    convert_csv()