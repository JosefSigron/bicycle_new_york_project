import csv
import sys
import os


def psv_to_csv(input_file, output_file=None):
    """
    Convert a pipe-separated values (PSV) file to a comma-separated values (CSV) file.
    
    Args:
        input_file (str): Path to the input PSV file
        output_file (str, optional): Path to the output CSV file. If not provided,
                                    will use the same name as input file but with .csv extension.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return False
    
    if output_file is None:
        # Replace .psv with .csv, or just append .csv if no .psv extension
        if input_file.lower().endswith('.psv'):
            output_file = input_file[:-4] + '.csv'
        else:
            output_file = input_file + '.csv'
    
    try:
        with open(input_file, 'r', newline='') as psv_file, \
             open(output_file, 'w', newline='') as csv_file:
            
            # Read from PSV file with pipe delimiter
            psv_reader = csv.reader(psv_file, delimiter='|')
            
            # Write to CSV file with comma delimiter
            csv_writer = csv.writer(csv_file)
            
            # Copy all rows from PSV to CSV
            for row in psv_reader:
                csv_writer.writerow(row)
        
        print(f"Successfully converted '{input_file}' to '{output_file}'")
        return True
    
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def main():
    # Check if correct number of arguments is provided
    if len(sys.argv) < 2:
        print("Usage: python src/psv_to_csv.py <input_psv_file> [output_csv_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Optional output file
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = psv_to_csv(input_file, output_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 