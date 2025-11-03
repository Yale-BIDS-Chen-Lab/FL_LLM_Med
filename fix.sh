NER_json_dir=${HOME}/NER_data
RE_json_dir=${HOME}/RE_data
for file in ${NER_json_dir}/**/*.json; do
  python3 src/data_fix/add_missing_col.py --file $file
done
for file in ${RE_json_dir}/*.json; do
  python3 src/data_fix/add_missing_col.py --file $file
done
