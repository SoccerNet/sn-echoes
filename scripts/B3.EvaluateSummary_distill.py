# with open("rouge_event_caption_asr.json", "w") as f: json.dump(rouge_event_caption_asr, f)
# with open("rouge_event_asr.json", "w") as f: json.dump(rouge_event_asr, f)
# with open("rouge_asr.json", "w") as f: json.dump(rouge_asr, f)
# with open("bert_event_caption_asr.json", "w") as f: json.dump(bert_event_caption_asr, f)
# with open("bert_event_asr.json", "w") as f: json.dump(bert_event_asr, f)
# with open("bert_asr.json", "w") as f: json.dump(bert_asr, f)

import json
import pandas as pd
bert_scores ={
    "event" :json.load(open("1bert_event_caption_asr.json")),
    "asr" :json.load(open("1bert_event_asr.json")),
    # "asr" :json.load(open("bert_asr.json"))
}
rouge_scores ={
    "event" :json.load(open("1rouge_event_caption_asr.json")),
    "asr" :json.load(open("1rouge_event_asr.json")),
    # "asr" :json.load(open("rouge_asr.json"))
}

# meteor_score={
#     "event_caption_asr" :json.load(open("meteor_event_caption_asr_nltk.json")),
#     "event_asr" :json.load(open("meteor_event_asr_nltk.json")),
#     "asr" :json.load(open("meteor_asr_nltk.json"))
# }


# filter  if keys in lang_csv
# lang_csv= pd.read_csv("/home/sushant/D1/DataSets/SoccerNet-Echoes/SN-echoes-lang.csv")
# all_games = sorted(set([e[:-2] for e in lang_csv[lang_csv.Lang=="en"]['Game/Half'].values.tolist()]))
# for key in bert_scores.keys(): bert_scores[key] = {k: v for k, v in bert_scores[key].items() if k in all_games}
# for key in rouge_scores.keys():
#     rouge_scores[key] = {k: v for k, v in rouge_scores[key].items() if  k in all_games}
# for key in meteor_score.keys():
#     meteor_score[key] = {k: v for k, v in meteor_score[key].items() if k in all_games}

import numpy as np

# print bert f1
print("bert f1", {key: np.mean([v[2][0] for v in value.values()]) for key, value in bert_scores.items()})
# breakpoint()
# filter lang==en
# breakpoint()
# breakpoint()
# print("meteor score")
# for key, value in meteor_score.items(): print(f"Meteor: {key}: {np.mean([v for v in value.values()])}")

import matplotlib.pyplot as plt
import os

# plt.violinplot([[v for v in value.values()] for key, value in meteor_score.items() ], showmeans=False, showmedians=False,showextrema=False)
# plt.boxplot([[v for v in value.values()] for key, value in meteor_score.items()], showfliers=False)
# plt.xticks([1, 2, 3], ["Event + Caption +ASR", "Event + ASR", "Only ASR"])
# plt.title('Meteor Score Violin Plot')
# plt.xlabel('Input for summarization')
# plt.ylabel('Meteor Score')
# # s asis from 0.30
# plt.ylim(0.20, 0.40)
# plt.tight_layout()
# plt.savefig("meteor_score.png", dpi=300, bbox_inches='tight')
# plt.show()
# exit(0)
# plot all values as different box plots


# breakpoint()

# for key, value in meteor_score.items(): print(f"Rouge1-Precision: {key}: {np.mean([v['rouge1'][0] for v in value.values()])}")

# print("rouge1")
# for key, value in rouge_scores.items(): print(f"Rouge1-Precision: {key}: {np.mean([v['rouge1'][0] for v in value.values()])}")
# for key, value in rouge_scores.items(): print(f"Rouge1-Recall: {key}: {np.mean([v['rouge1'][1] for v in value.values()])}")
# for key, value in rouge_scores.items(): print(f"RougeL-F1: {key}: {np.mean([v['rougeL'][2] for v in value.values()])}")


plt.rcParams.update({'font.size': 14})  # You can change the font size as needed

# breakpoint()

# volion plot for BERT F1
# breakpoint()
plt.violinplot([[v[2][0] for v in value.values()] for key, value in bert_scores.items() ], showmeans=False, showmedians=False,showextrema=False)
plt.boxplot([[v[2][0] for v in value.values()] for key, value in bert_scores.items()], showfliers=False)
plt.xticks([1, 2], ["Event", "ASR",])
plt.title('BERT Score Violin Plot')
plt.xlabel('Input for summarization')
plt.ylabel('BERT Score')
# s asis from 0.30
plt.ylim(0.82, 0.89)
plt.tight_layout()
plt.gcf().set_size_inches(4, 10)
plt.savefig("xbert_score.png", dpi=300, bbox_inches='tight')
plt.show()
plt.clf()


# breakpoint()
# volion plot for rouge1

# plt.violinplot([[v['rouge1'][2] for v in value.values()] for key, value in rouge_scores.items() ], showmeans=False, showmedians=False,showextrema=False)
# plt.boxplot([[v['rouge1'][2] for v in value.values()] for key, value in rouge_scores.items()], showfliers=False)
# plt.xticks([1, 2], ["Event", "ASR",])
# plt.title('Rouge1 Score Violin Plot')
# plt.xlabel('Input for summarization')
# plt.ylabel('Rouge1 Score')
# # s asis from 0.30
# plt.ylim(0.45, 0.65)
# plt.tight_layout()
# plt.gcf().set_size_inches(4, 10)
# plt.savefig("rouge1.png", dpi=300, bbox_inches='tight')
# plt.show()
# plt.clf()

# volion plot for rouge2
plt.violinplot([[v['rouge2'][2] for v in value.values()] for key, value in rouge_scores.items() ], showmeans=False, showmedians=False,showextrema=False)
plt.boxplot([[v['rouge2'][2] for v in value.values()] for key, value in rouge_scores.items()], showfliers=False)
plt.xticks([1, 2], ["Event", "ASR",])
plt.title('Rouge2 Score Violin Plot')
plt.xlabel('Input for summarization')
plt.ylabel('Rouge2 Score')
# s asis from 0.30
# plt.ylim(0.05, 0.19)
plt.tight_layout()
plt.gcf().set_size_inches(6, 10)
plt.savefig("xrouge2.png", dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

# volion plot for rougeL
plt.rcParams.update({'font.size': 14})  # You can change the font size as needed
plt.violinplot([[v['rougeL'][2] for v in value.values()] for key, value in rouge_scores.items() ], showmeans=False, showmedians=False,showextrema=False)
plt.boxplot([[v['rougeL'][2] for v in value.values()] for key, value in rouge_scores.items()], showfliers=False)
plt.xticks([1, 2], ["Event", "ASR",])
plt.title('rougeL Score Violin Plot')
plt.xlabel('Input for summarization')
plt.ylabel('RougeL Score')
# s asis from 0.30
# plt.ylim(0.115, 0.22)
plt.tight_layout()
# plt.figure(figsize=(8, 6)) 
plt.gcf().set_size_inches(4, 10)
plt.savefig("xrougeL.png", dpi=300, bbox_inches='tight')
plt.show()
plt.clf()


# violin plot for rougeLsum
# plt.rcParams.update({'font.size': 14})  # You can change the font size as needed
# plt.violinplot([[v['rougeLsum'][2] for v in value.values()] for key, value in rouge_scores.items() ], showmeans=False, showmedians=False,showextrema=False)
# plt.boxplot([[v['rougeLsum'][2] for v in value.values()] for key, value in rouge_scores.items()], showfliers=False)
# plt.xticks([1, 2], ["Event", "ASR",])
# plt.title('RougeLsum Score Violin Plot')
# plt.xlabel('Input for summarization')
# plt.ylabel('RougeLsum Score')
# # s asis from 0.30
# plt.ylim(0.30, 0.55)
# plt.tight_layout()
# plt.gcf().set_size_inches(6, 10)
# plt.savefig("xrougeLsum.png", dpi=300, bbox_inches='tight')
# plt.show()
# plt.clf()


print("rougeL")
for key, value in rouge_scores.items(): print(f"Rouge1-Precision: {key}: {np.mean([v['rougeL'][0] for v in value.values()])}")
for key, value in rouge_scores.items(): print(f"Rouge1-Recall: {key}: {np.mean([v['rougeL'][1] for v in value.values()])}")
for key, value in rouge_scores.items(): print(f"Rouge1-Recall: {key}: {np.mean([v['rougeL'][1] for v in value.values()])}")

print("rouge2")
for key, value in rouge_scores.items(): print(f"Rouge2-Precision: {key}: {np.mean([v['rouge2'][0] for v in value.values()])}")
for key, value in rouge_scores.items(): print(f"Rouge2-Recall: {key}: {np.mean([v['rouge2'][1] for v in value.values()])}")
for key, value in rouge_scores.items(): print(f"Rouge2-F1: {key}: {np.mean([v['rouge2'][2] for v in value.values()])}")

print("rougeLsum")
for key, value in rouge_scores.items(): print(f"RougeLsum-Precision: {key}: {np.mean([v['rougeLsum'][0] for v in value.values()])}")
for key, value in rouge_scores.items(): print(f"RougeLsum-Recall: {key}: {np.mean([v['rougeLsum'][1] for v in value.values()])}")
for key, value in rouge_scores.items(): print(f"RougeLsum-F1: {key}: {np.mean([v['rougeLsum'][2] for v in value.values()])}")

# breakpoint()




## for count plot



# import matplotlib.pyplot as plt
# import numpy as np

# print("input")
# import numpy as np
# input_count={}
# for column in ['ASR', 'Event+ASR', 'Event+Caption', 'Event+Caption+ASR']:
#     count_ = np.mean([len(x.replace("#", '').replace("\n", '').split()) for x in df_pivot[column]])
#     print(f"Count of words in {column}: {count_}")


# df_pivot_o = combiner_df.pivot(index='file', columns='type', values='output').reset_index()
# df_pivot_o.columns = [''.join(col).strip() for col in df_pivot_o.columns.values]
# df_pivot_o = df_pivot_o.rename(columns={'file_': 'file'})

# print("output")
# for column in ['ASR', 'Event+ASR', 'Event+Caption', 'Event+Caption+ASR']:
#     count_ =  np.mean([len(x.replace("#", ' ').replace("\n", ' ').split()) for x in df_pivot_o[column]])
#     print(f"Count of words in {column}: {count_}")

# breakpoint()





# # Data
# input_counts = {
#     "Event": 1020.7165,
#     "Event+Caption":  1013.675115207,
#     "ASR": 974.66359447,
# }

# output_counts = {
#     "Event":  1360.532258,
#     "Event+Caption": 3290.07603,
#     "ASR": 13861.253456221,
# }
# # Labels and values
# labels = list(input_counts.keys())
# input_values = list(input_counts.values())
# output_values = list(output_counts.values())

# x = np.arange(len(labels))

# # Plotting
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# # Input subplot
# ax1.bar(x, input_values, color='blue', alpha=0.7)
# ax1.set_ylabel('Word Count')
# ax1.set_title('Average Input Words')
# ax1.set_xticks(x)
# ax1.bar_label(ax1.containers[0])
# ax1.set_yscale('log')
# ax1.set_yticks([100, 1000, 10000, 16000])
# # more ticks
# ax1.set_xticklabels(labels)

# # Output subplot
# ax2.bar(x, output_values, color='green', alpha=0.7)
# ax2.set_ylabel('Word Count')
# ax2.set_title('Average Output Words')
# # show values on the top
# ax2.bar_label(ax2.containers[0])
# ax2.set_xticks(x)
# # ax2.set_yscale('log')
# ax2.set_yticks([500, 800, 1000, 1100])
# from matplotlib.ticker import ScalarFormatter
# # use number instad of scientific notation
# ax2.yaxis.set_major_formatter(ScalarFormatter())
# ax2.set_xticklabels(labels)

# plt.xlabel('Modality')
# plt.tight_layout()
# plt.rcParams.update({'font.size': 14})  # You can change the font size as needed
# plt.gcf().set_size_inches(6, 10)
# plt.show()
# plt.savefig("count.png", dpi=300, bbox_inches='tight')