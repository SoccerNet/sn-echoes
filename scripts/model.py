import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


class MultimodalClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes=-10, audio_model_=False, video_model_=False, text_model_=False):
        super(MultimodalClassifier, self).__init__()
        self.audio_model_ = audio_model_
        self.video_model_ = video_model_
        self.text_model_ = text_model_
        combined_hidden_size =  sum([audio_model_, video_model_, text_model_]) * 768
        self.fc = nn.Linear(combined_hidden_size, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        # print(f"Audio: {self.audio_model_.config.hidden_size if self.audio_model_ else 0}, Video: {self.video_model_.config.hidden_size if self.video_model_ else 0}, Text: {self.text_model_.config.hidden_size if self.text_model_ else 0}", end=" -> ")
        print("Combined: ", combined_hidden_size)
        
    def forward(self, text_input=None, audio_input=None, video_input=None, labels=None):
        audio_features, video_features, text_features = None, None, None
    
        if self.audio_model_:
            audio_features = audio_input
            
        if self.video_model_:
            video_features = video_input
            
        if self.text_model_:
            text_features = text_input

        combined_features = torch.cat([feature for feature in [audio_features, video_features, text_features] if feature is not None], dim=1) if any(feature is not None for feature in [audio_features, video_features, text_features]) else None

        output = self.fc(combined_features)
        loss = F.cross_entropy(output, labels) if labels is not None else None
        return {
            "outputs": output,
            "loss": loss
        }
