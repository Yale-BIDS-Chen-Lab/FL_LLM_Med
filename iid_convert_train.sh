NER_bio_dir=/home/cyy/NER_bio/train
NER_json_dir=/home/cyy/NER_data_train_iid
sub_dirs="MIMIC3:UTP:MTSamples"
iid_number=2

python3 src/command/iob_iid_split.py --data_dir="${NER_bio_dir}" --sub_dirs="${sub_dirs}" --skip_empty=1 --split_number=${iid_number} --output_dir="${NER_json_dir}"
