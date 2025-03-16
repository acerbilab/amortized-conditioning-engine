#!/bin/sh

# Function to run Python scripts based on input
run_script() {
    # $1 is the script filename, $2 is whether to save plots
    echo "Running $1 with save_plots=$2..."
    python $1 --save_plots $2
}

# Default to not save plots unless specified
save_plots=false

# Check for 'save' option
if [ "$1" = "save" ]; then
    save_plots=true
    shift  # Remove the first argument and process the rest
fi

# Define script names in a list
scripts=("classification.py" "kernel_identification.py" "latent_conditioning.py")

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    echo "No file options provided. Running all files."
    for script in "${scripts[@]}"; do
        run_script $script $save_plots
    done
else
    # Loop through all arguments and process each
    for arg in "$@"
    do
        case $arg in
            classification)
                run_script "classification.py" $save_plots
                ;;
            kernel_identification)
                run_script "kernel_identification.py" $save_plots
                ;;
            latent_conditioning)
                run_script "latent_conditioning.py" $save_plots
                ;;
            all)
                for script in "${scripts[@]}"; do
                    run_script $script $save_plots
                done
                break
                ;;
            *)
                echo "Unknown file option: $arg"
                echo "Valid options are: classification, kernel_identification, latent_conditioning, all"
                ;;
        esac
    done
fi
