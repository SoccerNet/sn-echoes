import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from model import MultimodalClassifier
import os
import os
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import classification_report

import wandb
from sklearn.metrics import confusion_matrix
import argparse

args = argparse.ArgumentParser()
args.add_argument("--audio_model", action='store_true')
args.add_argument("--video_model", action='store_true')
args.add_argument("--text_model", action='store_true')
args.add_argument("--lr", type=float, default=9e-3)
args.add_argument("--epochs", type=int, default=100)
args.add_argument("--batch_size", type=int, default=512)

args = args.parse_args()
print(args)

# audio_model = AutoModel.from_pretrained("facebook/wav2vec2-base").to("cuda")
# video_model = AutoModel.from_pretrained("MCG-NJU/videomae-base").to("cuda")
# text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# text_model = AutoModel.from_pretrained("distilbert-base-uncased").to("cuda")
# # audio_model.requires_grad_(False)
# # video_model.requires_grad_(False)
# # text_model.requires_grad_(False)


# text_input = ["Example input_values"," describing the ","audio segment","Example input_values ","describing the ","audio segmen","audio segment","Example input_values ","describing the ","audio segmen"]
# text_input=[torch.rand(768)]*500
# audio_input=[torch.rand(768) ]*500
# video_input=[torch.rand(768)]*500

os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_PROJECT"] = "SN-Echoes-MultiModal"
os.environ["WANDB_WATCH"]="false"
# os.environ['WANDB_DISABLED'] = 'true'


class CustomDataset(Dataset):
    def __init__(self, folder, df):
        self.class_names = ['BallOut', 'Card', 'Clearance', 'Corner', 'Foul', 'FreeKick', 'GoalAttempt', 'Substitution']
        self.label2id = {label: id for id, label in enumerate(self.class_names)}
        self.base_path = folder
        self.labels = []
        self.all_index = []
        for idx, row in df.iterrows():
            if os.path.exists(self.base_path + "text/" + str(idx) + ".npy"):
                self.labels.append(self.label2id[row['label']])
                self.all_index.append(idx)
        print("Total Valid Samples: ", len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cur_index = self.all_index[idx]
        # read from file
        text = np.load(self.base_path + "text/" + str(cur_index) + "_SportsBERT.npy")
        audio = np.load(self.base_path + "audio/" + str(cur_index) + ".npy")
        video = np.load(self.base_path + "video/" + str(cur_index) + ".npy")
        label = self.labels[idx]
        return {'text_input': text, 'audio_input': audio, 'video_input': video, 'labels': label}

df =pd.read_csv("/home/sushant/D1/DataSets/SoccerNet-Echoes/SN-echoes-class-v0.csv").set_index('id')
dataset = CustomDataset(df=df,  folder="/global/D1/projects/HOST/Datasets/SN-echoes-class-v0_features/")

train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=dataset.labels, random_state=42)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)
print("length of train dataset: ", len(train_dataset), ", validation dataset: ", len(test_dataset))

model = MultimodalClassifier(num_classes=len(dataset.class_names), audio_model_=args.audio_model, video_model_=args.video_model, text_model_=args.text_model).to("cuda")
print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

model_poxtfic={"a":args.audio_model, "v":args.video_model, "t":args.text_model}
model_name = "-".join([key for key, value in model_poxtfic.items() if value])

training_args = TrainingArguments(
    learning_rate=args.lr,  #0.000625
    weight_decay=0.001,
    output_dir='./results_SportsBERT/'+ model_name,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb"
)

temp_storage={}
def compute_metrics(eval_pred):
    logits_, labels = eval_pred
    logits = logits_.argmax(axis=1)
    accuracy = accuracy_score(labels, logits)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, logits, average='weighted',zero_division=0)
    conf_mat = confusion_matrix(y_true=labels, y_pred=logits, )
    temp_storage['labels'] = labels
    temp_storage['predictions'] = logits
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'conf_mat': conf_mat.tolist(),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]
)

trainer.train()

output = trainer.evaluate()
predictions, labels = temp_storage['predictions'], temp_storage['labels']
print(classification_report(labels, predictions, target_names=dataset.class_names, zero_division=0))
print("Final Confusion Matrix: ")
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=predictions, class_names=dataset.class_names)})
print(confusion_matrix(labels, predictions))
wandb.finish()

# python scripts/train_hf.py --audio_model 0 --video_model 0 --text_model 0 