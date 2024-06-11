import glob
import pandas as pd
import json
import os
# from summ_eval.summa_qa_metric import SummaQAMetric
from nltk.translate import meteor
from nltk import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score
rouge_ = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)



# import spacy
# import textdescriptives as td
# nlp = spacy.load("en_core_web_lg")
# nlp.add_pipe("textdescriptives/all")

# !pip install rouge-score bert_score  summ-eval transformers ==2.2.0 numpy ==1.11.0 bert-score ==0.3.13 
# pip install summ-eval transformers numpy bert-score==0.3.13  rouge-score 
# 3.9.1 numpy==1.19.1 pandas==1.0.4
#conda activate SN-echoes-eval
#conda create -n SN-echoes-eval  python=3.8

summary_dir="/home/sushant/D1/DataSets/SoccerNet-Echoes/Dataset_summary/"
# Glob patterns
patterns = [
    ("Summary1_caption_event.json", "Event+Caption"), ## Caption+ Event; ground truth
    ("Summary_caption_event_asr.json", "Event+Caption+ASR"),  # ASR
    ("Summary_caption_event.json", "Event+ASR"), ## EVENT
    ("Summary_asr.json", "ASR"),  ### NOT USED
]

# Initialize an empty list to store data
data = []

# Iterate over each pattern
for pattern, file_type in patterns:
    files = glob.glob(os.path.join(summary_dir, "**", pattern), recursive=True)
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content= f.read()
            try:
                content = json.loads(content)
            except:
                breakpoint()
            data.append({
                "type": file_type,
                "file": "/".join(file.split("/")[-4:-1]),
                "input": content["input"],
                "output": content["output"]
            })
combiner_df = pd.DataFrame(data)
# sort by file
combiner_df = combiner_df.sort_values(by="file")
combiner_df['input_count'] = [len(x.split()) for x in combiner_df['input']]

print(combiner_df.head())
df_pivot = combiner_df.pivot(index='file', columns='type', values='output').reset_index()
df_pivot.columns = [''.join(col).strip() for col in df_pivot.columns.values]
df_pivot = df_pivot.rename(columns={'file_': 'file'})
# for columns 'ASR', 'Event+ASR', 'Event+Caption', 'Event+Caption+ASR' count the number of words

df_pivot= df_pivot.dropna()
print("output")
import numpy as np
input_count={}
for column in ['Event+ASR', 'Event+Caption', 'Event+Caption+ASR']:
    count_ = np.mean([len(x.replace("#", '').replace("\n", '').split()) for x in df_pivot[column]])
    print(f"Count of words in {column}: {count_}")


df_pivot_o = combiner_df.pivot(index='file', columns='type', values='input').reset_index()
df_pivot_o.columns = [''.join(col).strip() for col in df_pivot_o.columns.values]
df_pivot_o = df_pivot_o.rename(columns={'file_': 'file'})
df_pivot_o= df_pivot_o.dropna()

print("input")
for column in ['Event+ASR', 'Event+Caption', 'Event+Caption+ASR']:
    count_ =  np.mean([len(x.replace("#", ' ').replace("\n", ' ').split()) for x in df_pivot_o[column]])
    print(f"Count of words in {column}: {count_}")


# df_pivotx= df_pivot.copy()
import numpy as np
# qa_result = qa_eval.evaluate_batch(df_pivot['Event+Caption+ASR'], df_pivot['Event+Caption'], aggregate=False)

# bert_results = bert_score(df_pivot['Event+Caption'].values.tolist(), df_pivot['Event+Caption+ASR'].values.tolist(), lang="en")  # bert_results[0].mean() 
#Event+Caption+ASR: 0.8472 , Event+ASR: 0.8475 F: 0.8411 , ASR: 

df_pivot= df_pivot.dropna()
breakpoint()

##ROUGE
rouge_event_caption_asr = {row['file']: rouge_.score(row['Event+Caption'], row['Event+Caption+ASR']) for index, row in df_pivot.iterrows()}
rouge_event_asr = {row['file']: rouge_.score(row['Event+Caption'], row['Event+ASR']) for index, row in df_pivot.iterrows()}
# rouge_asr = {row['file']: rouge_.score(row['Event+Caption'], row['ASR']) for index, row in df_pivot.iterrows()}
with open("rouge_event_caption_asr.json", "w") as f: json.dump(rouge_event_caption_asr, f)
with open("rouge_event_asr.json", "w") as f: json.dump(rouge_event_asr, f)

convert_to_numpy = lambda bert_asr: {key: tuple(map(lambda x: x.numpy().tolist(), value)) for key, value in bert_asr.items()}

bert_event_asr = convert_to_numpy({row['file']: bert_score([row['Event+Caption']], [row['Event+ASR']], lang="en") for index, row in df_pivot.iterrows()})
with open("1bert_event_asr.json", "w") as f: json.dump(bert_event_asr, f)

bert_asr = convert_to_numpy({row['file']: bert_score([row['Event+Caption']], [row['ASR']], lang="en") for index, row in df_pivot.iterrows()})
with open("bert_asr.json", "w") as f: json.dump(bert_asr, f)






#### MeteorMetric
# meteor_event_caption_asr = {row['file']:meteor([word_tokenize(row['Event+Caption'])], word_tokenize(row['Event+Caption+ASR'])) for index, row in df_pivot.iterrows()}
# with open("meteor_event_caption_asr_nltk.json", "w") as f: json.dump(meteor_event_caption_asr, f)
# print("Dumped meteor_event_caption_asr_nltk.json")
# meteor_event_asr = {row['file']:meteor([word_tokenize(row['Event+Caption'])], word_tokenize(row['Event+ASR'])) for index, row in df_pivot.iterrows()}
# with open("meteor_event_asr_nltk.json", "w") as f: json.dump(meteor_event_asr, f)
# print("Dumped meteor_event_asr_nltk.json")
# meteor_asr = {row['file']:meteor([word_tokenize(row['Event+Caption'])], word_tokenize(row['ASR'])) for index, row in df_pivot.iterrows()}
# with open("meteor_asr_nltk.json", "w") as f: json.dump(meteor_asr, f)
# print("Dumped meteor_asr_nltk.json")

## word 

# pymeteor.meteor(df_pivot.loc[1]['Event+Caption'], df_pivot.loc[1]['ASR'])

# df_pivot= df_pivot.head(10)
breakpoint()
convert_to_numpy = lambda bert_asr: {key: tuple(map(lambda x: x.numpy().tolist(), value)) for key, value in bert_asr.items()}

# print("dumping bert scores1")
# bert_event_caption_asr = convert_to_numpy({row['file']: bert_score([row['Event+Caption']], [row['Event+Caption+ASR']], lang="en") for index, row in df_pivot.iterrows()})
# with open("1bert_event_caption_asr.json", "w") as f: json.dump(bert_event_caption_asr, f)
print("dumping bert score 2")
bert_event_asr = convert_to_numpy({row['file']: bert_score([row['Event+Caption']], [row['Event+ASR']], lang="en") for index, row in df_pivot.iterrows()})
with open("1bert_event_asr.json", "w") as f: json.dump(bert_event_asr, f)

# bert_asr = convert_to_numpy({row['file']: bert_score([row['Event+Caption']], [row['ASR']], lang="en") for index, row in df_pivot.iterrows()})
breakpoint()
with open("bert_asr.json", "w") as f: json.dump(bert_asr, f)

exit()



# np.mean([e['rouge1'].fmeasure  for e in  rouge_results]) #0.5029

for index, row in df_pivot.iterrows():
    # have 'Event+ASR', 'Event+Caption', 'Event+Caption+ASR'
    gold_label = row['Event+Caption']
    breakpoint()

    # evaluate Event+ASR
    qa_result = qa_eval.evaluate_example(gold_label, row['Event+ASR'])
    print(f"QA Event+ASR: {qa_result}")
    rouge_result = rouge_.score(gold_label, row['Event+ASR'])
    print(f"Rouge Event+ASR: {rouge_result}")
    P, R, F1  = bert_score([row['Event+ASR']], [gold_label], lang="en")
    print(f"BertScore Event+ASR: {P, R, F1}")


breakpoint()

# calculate count
# combiner_df['input_count'] = [len(x) for x in combiner_df['input']]
# combiner_df['output_count'] = [len(x) for x in combiner_df['output']]
# breakpoint()



# combiner_df['flesch_reading_ease_ts'] = [textstat.flesch_reading_ease(x) for x in combiner_df['output']]
# combiner_df['flesch_kincaid_grade_ts'] = [textstat.flesch_kincaid_grade(x) for x in combiner_df['output']]
# combiner_df['smog_index_ts'] = [textstat.smog_index(x) for x in combiner_df['output']]
# combiner_df['coleman_liau_index_ts'] = [textstat.coleman_liau_index(x) for x in combiner_df['output']]
# combiner_df['automated_readability_index_ts'] = [textstat.automated_readability_index(x) for x in combiner_df['output']]
# combiner_df['dale_chall_readability_score_ts'] = [textstat.dale_chall_readability_score(x) for x in combiner_df['output']]
# combiner_df['difficult_words_ts'] = [textstat.difficult_words(x) for x in combiner_df['output']]
# combiner_df['linsear_write_formula_ts'] = [textstat.linsear_write_formula(x) for x in combiner_df['output']]
# combiner_df['gunning_fog_ts'] = [textstat.gunning_fog(x) for x in combiner_df['output']]
# combiner_df['text_standard_ts'] = [textstat.text_standard(x) for x in combiner_df['output']]
# combiner_df['fernandez_huerta_ts'] = [textstat.fernandez_huerta(x) for x in combiner_df['output']]
# combiner_df['szigriszt_pazos_ts'] = [textstat.szigriszt_pazos(x) for x in combiner_df['output']]
# combiner_df['gutierrez_polini_ts'] = [textstat.gutierrez_polini(x) for x in combiner_df['output']]
# combiner_df['crawford_ts'] = [textstat.crawford(x) for x in combiner_df['output']]
# combiner_df['gulpease_index_ts'] = [textstat.gulpease_index(x) for x in combiner_df['output']]
# combiner_df['osman_ts'] = [textstat.osman(x) for x in combiner_df['output']]

# docs = nlp.pipe(combiner_df.output.values)
# textdescriptives = td.extract_df(docs, include_text = False)
# # textdescriptives has same index as combiner_df
# combiner_df = pd.concat([combiner_df, textdescriptives], axis=1)
# # group by type and print describe
# # Iterate over each group

desc_df_list = []
for name, group in combiner_df.groupby("type"):
    # Remove the input and output columns type and file
    groupx = group.drop(columns=["type", "file", "input", "output"])
    # Describe the group and transpose the result to have it in a single row
    desc = groupx.describe().T
    # Add a column with the group type
    desc['Type'] = name
    # Append the description to the DataFrame
    desc_df_list.append(desc)
    
desc_df= pd.concat(desc_df_list)
breakpoint()
# sort by index
desc_df = desc_df.sort_index()
# remove count
desc_df = desc_df.drop(columns="count")
# bring type to start
desc_df = desc_df[["Type"] + [col for col in desc_df.columns if col != "Type"]]
desc_df.to_csv("group_descriptions.csv")
print("Dumped group descriptions to group_descriptions.csv")
# # HTML interactive table plot
# import pandas_bokeh
# from bokeh.plotting import figure, output_file, save
# desc_df['metric']= desc_df.index

# desc_df_bokeh= desc_df[["mean", "metric",  'Type']]
# # convert type to int encode

# p_bar = desc_df_bokeh.plot_bokeh.scatter(x="metric", y="mean", category="Type", sizing_mode="stretch_both", alpha=0.6,return_html=True)
# with open("group_descriptions.html", "w") as f:  f.write(p_bar)

# # to html
# p_bar.to_html("group_descriptions.html")

# breakpoint()
