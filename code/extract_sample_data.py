#!/usr/bin/env python
"""
This script queries a sample data CSV spreadsheet, extracts all
rows that in the indicated column contain the indicated string, and
writes the results to a new CSV file containing only the slice specific
data.

The output CSV contains the same columns as the input CSV as well as
the column 'sample_id', which is just the corresponding index in the
inoput CSV file.

Example
-------
python code/extract_sample_data.py data/plates_6_11_12_13_14_16_17_18.csv date 2019-03-22 test/sample_data.csv --show

"""

import pandas as pd

from pathlib import Path
from argparse import ArgumentParser, RawDescriptionHelpFormatter


if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("input_sample_data",           help="/path/to/input/sample_data.csv",                                     type=str)
    parser.add_argument("column",                      help="The column containing the query pattern.",                           type=str)
    parser.add_argument("pattern",                     help="Rows containing this pattern in the indicated column are exported.", type=str)
    parser.add_argument("output_sample_data",          help="/path/to/output/sample_data.csv",                                    type=str)
    parser.add_argument("--show", action="store_true", help="Print header of output data frame.")

    args = parser.parse_args()

    data = pd.read_csv(args.input_sample_data)
    subset = data[data[args.column] == args.pattern]
    subset.index.rename("sample_id", inplace=True)
    subset.reset_index(inplace=True)
    subset.to_csv(args.output_sample_data)

    if args.show:
        print(subset)
