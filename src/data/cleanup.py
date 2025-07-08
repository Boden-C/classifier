#!/usr/bin/env python3
"""
Cleanup script to remove duplicate rows in raw.csv where the 'func' column is duplicate.
Uses a hash table implementation for efficient duplicate detection.
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def remove_duplicates(input_file: Path, output_file: Path, key_column: str) -> Tuple[int, int]:
    """
    Remove duplicate rows from a CSV file based on a specific column.

    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
        key_column: Column name to check for duplicates

    Returns:
        Tuple containing (total_rows, unique_rows) counts
    """
    # Create directory for output file if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    seen_values: Set[str] = set()
    total_rows = 0
    unique_rows = 0

    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)

            if key_column not in reader.fieldnames:
                raise ValueError(f"Column '{key_column}' not found in CSV. Available columns: {reader.fieldnames}")

            with open(output_file, "w", encoding="utf-8", newline="") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()

                for row in reader:
                    total_rows += 1
                    key_value = row[key_column]

                    # Check if this value has been seen before
                    if key_value in seen_values:
                        continue

                    # First time seeing this value, add to set and write to output
                    seen_values.add(key_value)
                    writer.writerow(row)
                    unique_rows += 1

        return total_rows, unique_rows

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied when accessing '{input_file}' or '{output_file}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def main():
    """Entry point for the script."""
    # Define file paths
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent

    input_file = project_root / "src" / "data" / "raw.csv"
    output_file = project_root / "src" / "data" / "dataset.csv"
    key_column = "func"

    print(f"Removing duplicate rows from {input_file}")
    print(f"Based on column: '{key_column}'")

    total_rows, unique_rows = remove_duplicates(input_file, output_file, key_column)

    print(f"Total rows processed: {total_rows}")
    print(f"Unique rows kept: {unique_rows}")
    print(f"Duplicates removed: {total_rows - unique_rows}")
    print(f"Clean data saved to: {output_file}")


if __name__ == "__main__":
    main()
