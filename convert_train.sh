rm -rf /home/cyy/NER_data/train
mkdir -p /home/cyy/NER_data/train
for dir in /home/cyy/NER_bio/train/*; do
  python3 src/command/medical_iob_converter.py --data_dir=${dir} --output_file=$(basename ${dir}).json
  mv $(basename ${dir}).json /home/cyy/NER_data/train/
done
