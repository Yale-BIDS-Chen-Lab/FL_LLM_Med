RE_data_file=/home/cyy/RE_data/RE_MTSample_train.json
OUTPUT_json_dir=/home/cyy/iid_RE_data
iid_number=2

python3 src/command/json_iid_split.py --data_file="${RE_data_file}" --split_number=${iid_number} --output_dir="${OUTPUT_json_dir}"
