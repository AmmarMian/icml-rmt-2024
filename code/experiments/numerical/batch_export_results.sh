#!/bin/bash

# Function to process a directory
process_directory() {
    dir="$1"
    mean_file="${dir}/mean.csv"
    q5_file="${dir}/5.csv"
    q95_file="${dir}/95.csv"
    output_file="${dir}/output.tex"

    if [ -f "$mean_file" ] && [ -f "$q5_file" ] && [ -f "$q95_file" ]; then
        # Replace <your_script> with the actual script you want to execute
      python experiments/numerical/export_to_latex.py "$mean_file" "$q5_file" "$q95_file" "$output_file"
    fi
}

# Main script
main() {
    if [ "$#" -ne 1 ]; then
        echo "Usage: $0 <parent_directory>"
        exit 1
    fi

    parent_directory="$1"

    # Iterate through subdirectories
    for dir in "$parent_directory"/*/; do
        process_directory "$dir"
    done
}

# Run the main script
main "$@"

