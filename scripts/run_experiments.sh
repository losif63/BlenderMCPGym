#!/bin/bash

run_if_missing() {
    local name="$1" model="$2" prompt_type="$3"
    local stem="${name%.*}"
    if [ -d "recreation/${stem}/claudecode/${model}" ]; then
        echo "Skipping ${name} / ${model} (recreation/${stem}/claudecode/${model} already exists)"
    else
        python src/variance/run_claudecode.py --image "$name" --model "$model" --prompt-type "$prompt_type"
    fi
}

for filename in images/*; do
    name=$(basename "$filename")
    if [[ $name == *"advanced"* ]]; then
        # run_if_missing "$name" haiku-4.5 advanced
        # run_if_missing "$name" sonnet-4.6 advanced
        # run_if_missing "$name" opus-4.6 advanced
        run_if_missing "$name" opus-4.7 advanced
        # run_if_missing "$name" deepseek-v4-flash advanced
        # run_if_missing "$name" deepseek-v4-pro advanced
    elif [[ $name == *"beginner"* ]]; then
        # run_if_missing "$name" haiku-4.5 beginner
        # run_if_missing "$name" sonnet-4.6 beginner
        # run_if_missing "$name" opus-4.6 beginner
        run_if_missing "$name" opus-4.7 beginner
        # run_if_missing "$name" deepseek-v4-flash beginner
        # run_if_missing "$name" deepseek-v4-pro beginner
    fi
done