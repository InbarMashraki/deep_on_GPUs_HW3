#!/bin/bash

# Ensure the script exits if any command fails
set -e
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
MAIL_USER="inbar.m@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
# Define the output directory
OUTPUT_DIR="/home/inbar.m/hw/deep_on_GPUs_HW3/out"
NOTEBOOK_PATH="/home/inbar.m/hw/deep_on_GPUs_HW3//part1_training.sh"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Function to run the notebook
run_notebook(){
  NOTEBOOK_NAME="Part1_Sequence.ipynb"

  sbatch \
    -N $NUM_NODES \
    -c $NUM_CORES \
    --gres=gpu:$NUM_GPUS \
    --job-name "notebook_run" \
    --mail-user $MAIL_USER \
    --mail-type $MAIL_TYPE \
    -o "${OUTPUT_DIR}/notebook_run.out" \
    <<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB 'notebook_run' of '${NOTEBOOK_NAME}' STARTING ***"

# Run the notebook 
python -m main run-nb ${NOTEBOOK_PATH}

echo "*** SLURM BATCH JOB 'notebook_run' DONE ***"
EOF
}

# Call the function to run the notebook
run_notebook
