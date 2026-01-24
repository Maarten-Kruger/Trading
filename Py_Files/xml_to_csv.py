import xml.etree.ElementTree as ET
import csv
import sys
import os
import glob

def convert_xml_to_csv(xml_file):
    if not os.path.exists(xml_file):
        print(f"Error: File '{xml_file}' not found.")
        return

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML '{xml_file}': {e}")
        return

    # Namespace for Excel XML
    ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}

    # Find the worksheet
    worksheet = root.find(".//ss:Worksheet[@ss:Name='Tester Optimizator Results']", ns)
    if worksheet is None:
        # Fallback to first worksheet
        worksheet = root.find(".//ss:Worksheet", ns)

    if worksheet is None:
        print(f"Error: No worksheet found in '{xml_file}'.")
        return

    table = worksheet.find("ss:Table", ns)
    if table is None:
        print(f"Error: No table found in '{xml_file}'.")
        return

    rows = table.findall("ss:Row", ns)

    # Determine output filename
    base_name = os.path.splitext(xml_file)[0]
    csv_file = f"{base_name}.csv"

    print(f"Converting '{xml_file}' to '{csv_file}'...")

    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')

            for row in rows:
                csv_row = []
                cells = row.findall("ss:Cell", ns)

                # Handle ss:Index (sparse rows)
                current_index = 0

                for cell in cells:
                    # check for ss:Index
                    idx_attr = cell.get(f"{{{ns['ss']}}}Index")
                    if idx_attr:
                        target_index = int(idx_attr) - 1 # 1-based to 0-based
                        while current_index < target_index:
                            csv_row.append("")
                            current_index += 1

                    data = cell.find("ss:Data", ns)
                    value = ""
                    if data is not None:
                        value = data.text if data.text else ""

                        data_type = data.get(f"{{{ns['ss']}}}Type")

                        # Convert decimals for numbers
                        if data_type == 'Number':
                            value = value.replace('.', ',')
                        elif value and value.count('.') == 1 and value.replace('.', '').isdigit():
                            # heuristic for numbers marked as String or untyped
                             value = value.replace('.', ',')

                    csv_row.append(value)
                    current_index += 1

                writer.writerow(csv_row)

    except IOError as e:
        print(f"Error writing CSV file '{csv_file}': {e}")

def process_path(path):
    # Remove quotes if present
    if (path.startswith('"') and path.endswith('"')) or \
       (path.startswith("'") and path.endswith("'")):
        path = path[1:-1]

    path = path.strip()

    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist.")
        return

    if os.path.isdir(path):
        print(f"Processing all XML files in directory: {path}")
        # Find all xml files in the directory
        xml_files = glob.glob(os.path.join(path, "*.xml"))
        # Case insensitive search fallback if needed (on linux glob is case sensitive)
        if not xml_files:
             xml_files = [f for f in os.listdir(path) if f.lower().endswith('.xml')]
             xml_files = [os.path.join(path, f) for f in xml_files]

        if not xml_files:
            print("No XML files found in the directory.")
            return

        for xml_file in xml_files:
            convert_xml_to_csv(xml_file)

        print(f"Processed {len(xml_files)} files.")

    elif os.path.isfile(path):
        convert_xml_to_csv(path)
        print("Conversion complete.")
    else:
        print(f"Error: '{path}' is not a valid file or directory.")

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter the XML filename or folder path to convert: ").strip()

    process_path(path)

if __name__ == "__main__":
    main()
