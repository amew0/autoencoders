for input_file in "$@"
do  
    echo "Working on ${input_file} @ $(date)" 
    filename=$(basename "${input_file}" .yml)
    python -u eit.py ${input_file} > ./results/${filename}.std 2> ./results/${filename}.err
    echo "Done with ${input_file} @ $(date)"
done