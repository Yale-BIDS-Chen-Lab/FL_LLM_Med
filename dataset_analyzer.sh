NER_bio_dir=/home/cyy/NER_bio/train
sub_dirs="MIMIC3:UTP:MTSamples"

python3 src/command/iob_distribution_analyzer.py --data_dir="${NER_bio_dir}" --sub_dirs="${sub_dirs}"
