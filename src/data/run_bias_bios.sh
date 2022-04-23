
# creating the dataset folder
mkdir data/bert_encode_biasbios

# Encode text using Bert
for split in 'train' 'dev' 'test'
do
    python encode_bert_states.py \
        --input_file ../../data/biasbios/EN/$split.pickle \
        --output_dir ../../data/bert_encode_biasbios/EN/wo_gender/ \
        --split $split
done
