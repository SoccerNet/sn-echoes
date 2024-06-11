
# run script to generate script to crop videos and index file
python scripts/1.create_splits.py

# crop video [in h001]
cd /global/D1/projects/HOST/Datasets
xargs -a /home/sushant/D1/DataSets/SoccerNet-Echoes/SN-echoes-class-v0.txt -d '\n' -P 100 -I {} sh -c '{}'


# python read video chunks and create embedding for each modality
python 2.get_embeddings.py

# train model

cd /home/sushant/D1/DataSets/SoccerNet-Echoes/
conda activate NLPassign
python scripts/train_hf.py --audio_model --video_model --text_model