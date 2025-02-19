NER_bio_dir=/home/cyy/NER_bio
NER_json_dir=/home/cyy/NER_data
mkdir -p ${NER_json_dir}/train
for dir in ${NER_bio_dir}/train/*; do
  python3 src/command/medical_iob_converter.py --data_dir=${dir} --output_file=$(basename ${dir}).json
  mv $(basename ${dir}).json ${NER_json_dir}/train/
done


mkdir -p ${NER_json_dir}/test
for dir in ${NER_bio_dir}/test/*; do
  python3 src/command/medical_iob_converter.py --data_dir=${dir} --output_file=$(basename ${dir}).json
  mv $(basename ${dir}).json ${NER_json_dir}/test/
done
