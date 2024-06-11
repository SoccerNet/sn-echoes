import pandas as pd
from tqdm import tqdm
import os
import torchvision.io as io
from torchaudio.transforms import Resample
import numpy as np



video_path= "/global/D1/projects/HOST/Datasets/SN-echoes-class-v0/"
csv_path= "/home/sushant/D1/DataSets/SoccerNet-Echoes/SN-echoes-class-v0.csv"
features_path =  "/global/D1/projects/HOST/Datasets/SN-echoes-class-v0_features/"
audio_feat_path = features_path+"audio/"
video_feat_path = features_path+"video/"
text_feat_path = features_path+"text/"
os.makedirs(audio_feat_path, exist_ok=True)
os.makedirs(video_feat_path, exist_ok=True)
os.makedirs(text_feat_path, exist_ok=True)


read_lang = pd.read_csv(csv_path)
read_lang.set_index('id', inplace=True)

from transformers import AutoTokenizer, AutoModel
import torch
from transformers import AutoFeatureExtractor
from transformers import VideoMAEImageProcessor

audio_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
audio_model = AutoModel.from_pretrained("facebook/wav2vec2-base").to("cuda")

video_model = AutoModel.from_pretrained("MCG-NJU/videomae-base").to("cuda")
video_feature_extractor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
video_feature_extractor.do_center_crop = False


text_tokenizer = AutoTokenizer.from_pretrained("microsoft/SportsBERT")
text_model = AutoModel.from_pretrained("microsoft/SportsBERT").to("cuda")

audio_model.requires_grad_(False)
video_model.requires_grad_(False)
text_model.requires_grad_(False)

# image_mean= torch.tensor(video_feature_extractor.image_mean).to("cuda").reshape(1, 3, 1, 1)
# image_std= torch.tensor(video_feature_extractor.image_std).to("cuda").reshape(1, 3, 1, 1)

for idx, row in tqdm(read_lang.iterrows(), total=len(read_lang)):
    video_file = video_path + str(idx) + ".mp4"
    if not os.path.exists(video_file):
        print("❌---->", video_file)
        # load audio and video separately from the video file
        
        # continue
    else:
        # print("✅---->", video_file)
        text_feat_file_name = text_feat_path +str(idx)+"_SportsBERT.npy"
        if os.path.exists(text_feat_file_name):
            continue # already processed
        video, audio, info  = io.read_video(video_file, pts_unit='sec')

        print(video.shape) #[400, 224, 398, 3]
        print(audio.shape) #[2, 768000]
        print(info) # {'video_fps': 25.0, 'audio_fps': 48000} 
        resampler = Resample(orig_freq=int(info['audio_fps']), new_freq=16000)
        try:
            audio_resampled = resampler(audio).mean(dim=0).flatten()
            audio_features = audio_feature_extractor(audio_resampled, return_tensors="pt", sampling_rate=16000.).to("cuda")
            audio_output = audio_model(**audio_features).last_hidden_state.mean(dim=1).to("cpu").detach().flatten().numpy()  # 768

            # process video
            video_frames = video[::(len(video)//16)].permute(0, 3, 1, 2) # video_framesx.pixel_values.shape
            video_frames = torch.nn.functional.interpolate(video_frames, size=(224, 224), mode='bilinear', align_corners=False)
            video_framesx =  video_feature_extractor(list(video_frames), return_tensors="pt").to("cuda") # normalize the video as in the model
            video_output = video_model(**video_framesx).last_hidden_state.mean(dim=1).to("cpu").detach().flatten().numpy() 
            
            # process text
            text_features = text_tokenizer(row['asrs_txt'], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            text_output = text_model(**text_features).last_hidden_state.mean(dim=1).to("cpu").detach().flatten().numpy()

            # dump the embeddings as numpy arrays
            np.save(audio_feat_path +str(idx)+".npy", audio_output)
            np.save(video_feat_path +str(idx)+".npy", video_output)
            np.save(text_feat_file_name, text_output)
        except Exception as e:
            print("❌---->", video_file, e)
            continue
        # breakpoint()