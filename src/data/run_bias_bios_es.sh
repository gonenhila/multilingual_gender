
# creating the dataset folder

# Encode text using Bert
for split in 'train' 'dev' 'test'
do
    python encode_bert_states.py \
        --input_file ../../data/biasbios/ES/$split.pickle \
        --output_dir ../../data/bert_encode_biasbios/ES/wo_gender/ \
        --split $split
done
