RE_csv_dir=${HOME}/RE/RE
RE_json_dir=${HOME}/RE_data

mkdir -p ${RE_json_dir}
python3 src/command/medical_re_csv_parse.py --csv_files "${RE_csv_dir}/RE_train.csv ${RE_csv_dir}/RE_dev.csv" --output_file="${RE_json_dir}/train.json"

for csvfile in RE_i2b2_test RE_MIMIC3_test RE_MTSample_test RE_UTP_test; do
  python3 src/command/medical_re_csv_parse.py --csv_files "${RE_csv_dir}/${csvfile}.csv" --output_file="${RE_json_dir}/${csvfile}.json"
done
