import xml.etree.ElementTree as ET
import csv
import sys
import os

def convert_xml_to_csv(xml_file):
    if not os.path.exists(xml_file):
        print(f"Error: File '{xml_file}' not found.")
        return

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return

    # Namespace for Excel XML
    # The default namespace is usually the spreadsheet one too,
    # but we map it to 'ss' for find/findall usage.
    ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}

    # Find the worksheet
    # Try to find 'Tester Optimizator Results'
    worksheet = root.find(".//ss:Worksheet[@ss:Name='Tester Optimizator Results']", ns)
    if worksheet is None:
        # Fallback to first worksheet
        worksheet = root.find(".//ss:Worksheet", ns)

    if worksheet is None:
        print("Error: No worksheet found in the XML file.")
        return

    table = worksheet.find("ss:Table", ns)
    if table is None:
        print("Error: No table found in the worksheet.")
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

        print("Conversion complete.")

    except IOError as e:
        print(f"Error writing CSV file: {e}")

def main():
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    else:
        xml_file = input("Enter the XML filename to convert: ").strip()
        # Remove quotes if user added them
        if (xml_file.startswith('"') and xml_file.endswith('"')) or \
           (xml_file.startswith("'") and xml_file.endswith("'")):
            xml_file = xml_file[1:-1]

    convert_xml_to_csv(xml_file)

if __name__ == "__main__":
    main()
