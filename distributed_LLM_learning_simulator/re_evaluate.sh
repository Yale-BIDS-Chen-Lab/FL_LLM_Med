# export SESSION_DIR="session/adaptor_avg/hugging_face_yale_ner_iid/hugging_face_causal_lm_deepseek-ai/DeepSeek-R1-Distill-Llama-8B/2025-02-22_22_09_46/240613044987893863958513903783342351560"
test_dir=$HOME/RE_data/test/
for file in MIMIC3.json UTP.json MTSamples.json; do
  python3 ./src/command/NER_evaluator.py --session_dir="${SESSION_DIR}" --test_file="${test_dir}/${file}" --debug_file ./${file}.debug.txt --skipped_tags "problem treatment test drug"
done
