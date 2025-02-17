rm -rf /home/cyy/NER_data/test
mkdir -p /home/cyy/NER_data/test
for dir in /home/cyy/NER_bio/test/*; do
  python3 src/command/medical_iob_converter.py --data_dir=${dir} --output_file=$(basename ${dir}).json
  mv $(basename ${dir}).json /home/cyy/NER_data/test/
done
