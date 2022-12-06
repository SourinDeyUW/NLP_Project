# NLP_Project



cd codes

Then run the preparing training data.ipynb file. 

Then run the prepare test.ipynb file.

Then git clone https://github.com/dmis-lab/biobert-pytorch.git

    
Then to download the dataset type:

./download.sh

As a next step , you need to put the train.tsv and test.tsv file created by the notebooks inside biobert-pytorch/datasets/NER/BC4CHEMD.

BC4CHEMD is the dataset that is most suited and we manually merged our own dataset with it.


Now cd to biobert-pytorch/named-entity-recognition.

Now, remove all dataset names except BC4CHEMD in the preprocess.sh .

Then run the following in batch file such as submit.sh .

export DATA_DIR=../datasets/NER

export ENTITY=BC4CHEMD

python run_ner.py \

    --data_dir ${DATA_DIR}/${ENTITY} \
    
    --labels ${DATA_DIR}/${ENTITY}/labels.txt \
    
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    
    --output_dir output/${ENTITY} \
    
    --max_seq_length 128 \
    
    --num_train_epochs 5 \
    
    --per_device_train_batch_size 64 \
    
    --save_steps 5000 \
    
    --seed 1 \
    
    --do_train \
    
    --do_eval \
    
    --do_predict \
    
    --output_dir output_version_final



Now the result file is the predicted_result.csv .

This file is analyzed using the notebook named: prepare result.ipynb
