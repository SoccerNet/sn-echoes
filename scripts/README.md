---
dataset_info:
- config_name: whisper_v1
  features:
  - name: segment_index
    dtype: int32
  - name: start_time
    dtype: float32
  - name: end_time
    dtype: float32
  - name: text
    dtype: string
  - name: game
    dtype: string
  splits:
  - name: train
    num_bytes: 108020559
    num_examples: 780160
  download_size: 113870675
  dataset_size: 108020559
- config_name: whisper_v1_en
  features:
  - name: segment_index
    dtype: int32
  - name: start_time
    dtype: float32
  - name: end_time
    dtype: float32
  - name: text
    dtype: string
  - name: game
    dtype: string
  splits:
  - name: train
    num_bytes: 74527818
    num_examples: 563064
  download_size: 113870675
  dataset_size: 74527818
- config_name: whisper_v2
  features:
  - name: segment_index
    dtype: int32
  - name: start_time
    dtype: float32
  - name: end_time
    dtype: float32
  - name: text
    dtype: string
  - name: game
    dtype: string
  splits:
  - name: train
    num_bytes: 105607800
    num_examples: 761240
  download_size: 113870675
  dataset_size: 105607800
- config_name: whisper_v2_en
  features:
  - name: segment_index
    dtype: int32
  - name: start_time
    dtype: float32
  - name: end_time
    dtype: float32
  - name: text
    dtype: string
  - name: game
    dtype: string
  splits:
  - name: train
    num_bytes: 71273737
    num_examples: 537526
  download_size: 113870675
  dataset_size: 71273737
- config_name: whisper_v3
  features:
  - name: segment_index
    dtype: int32
  - name: start_time
    dtype: float32
  - name: end_time
    dtype: float32
  - name: text
    dtype: string
  - name: game
    dtype: string
  splits:
  - name: train
    num_bytes: 120436746
    num_examples: 923181
  download_size: 113870675
  dataset_size: 120436746
- config_name: whisper_v3_en
  features:
  - name: segment_index
    dtype: int32
  - name: start_time
    dtype: float32
  - name: end_time
    dtype: float32
  - name: text
    dtype: string
  - name: game
    dtype: string
  splits:
  - name: train
    num_bytes: 84488589
    num_examples: 679738
  download_size: 113870675
  dataset_size: 84488589
---

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