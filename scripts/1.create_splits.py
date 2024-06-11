import glob
import os
import json
import pandas as pd
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm

read_lang = pd.read_csv("SN-echoes-lang.csv")
base_path = "Dataset/"

read_lang = read_lang[read_lang.iloc[:, 1] == 'en']
# have count of third column
# read_lang.iloc[:, 2].value_counts()


def remove_repeating_entries(json_data):
    cleaned_data = {}
    prev_text = None

    for key, value in json_data.items():
        start, end, text = value
        if text.strip() != prev_text:
            cleaned_data[key] = value
        prev_text = text.strip()

    return cleaned_data


event_to_group = {'Ball out of play': 'BallOut', 
'Throw-in': 'BallOut', 
'Yellow card': 'Card',
 'Yellow->red card': 'Card', 
 'Red card': 'Card', 
 'Clearance': 'Clearance', 
 'Corner': 'Corner', 
 'Foul': 'Foul',
   'Offside': 'Foul',
'Penalty': 'FreeKick',
 'Direct free-kick': 'FreeKick', 
 'Indirect free-kick': 'FreeKick',
  'Shots on target': 'GoalAttempt',
   'Shots off target': 'GoalAttempt', 
   'Goal': 'GoalAttempt', 
   'Substitution':  'Substitution'}

print({id:label for id, label in  enumerate(sorted(set(event_to_group.values())))})

timeDelta = 10  # minimum time gap before and after the events selected
asr_delta = 15  # ASR starts and ends within asr_delta seconds of the event
word_threshold = 10  # number of unique words per event in ASR


all_df = []
# read_lang= read_lang.head(10) #######
for idx, row in tqdm(read_lang.iterrows(), total=len(read_lang)):
    json_path = base_path+"whisper_" + \
        row['selected']+"/" + row['Game/Half']+"_asr.json"
    json_data = json.load(open(json_path))['segments']
    cleaned_data = remove_repeating_entries(json_data)
    cleaned_data_df = pd.DataFrame(cleaned_data.values(), columns=[
                                   "start", "end", "text"])
    # print(f"{json_path}: {len(json_data)},   {len(cleaned_data)},  {len(json_data) - len(cleaned_data)}")
    Labels_v2_path = "SoccerNet_224p/" + row['Game/Half'][:-1]+"Labels-v2.json"
    # check if path exists
    if not os.path.exists(Labels_v2_path):
        print("❌---->", Labels_v2_path)
        continue
    Labels_v2 = json.load(open(Labels_v2_path))['annotations']
    Labels_v2 = [e for e in Labels_v2 if e['gameTime'].startswith(
        row['Game/Half'][-1])]
    for e in Labels_v2:
        times = e['gameTime'][4:].split(":")
        e['gameTime'] = int(times[0])*60 + int(times[1])
    Labels_v2_df = pd.DataFrame(Labels_v2)
    Labels_v2_df['time_diff_before'] = Labels_v2_df['gameTime'].diff().shift(-1)
    Labels_v2_df['time_diff_after'] = Labels_v2_df['gameTime'].diff()
    Labels_v2_df = Labels_v2_df[Labels_v2_df['visibility'] == "visible"]
    filtered_df = Labels_v2_df[(Labels_v2_df['time_diff_before'] >= timeDelta) & (
        Labels_v2_df['time_diff_after'] >= timeDelta)]
    filtered_df = filtered_df.copy()
    # filtered_df['asrs'] = filtered_df['gameTime'].apply(lambda x: cleaned_data_df.loc[(cleaned_data_df['start'] >= (
    #     x - asr_delta)) & (cleaned_data_df['end'] > x) & (cleaned_data_df['start'] <= (x + asr_delta))].index.tolist())
    # breakpoint()
    filtered_df['asrs'] = filtered_df['gameTime'].apply(lambda x:  cleaned_data_df.loc[(cleaned_data_df['start'] >= (x - asr_delta))  & (cleaned_data_df['end'] < x)].index.tolist() + [-1] + cleaned_data_df.loc[(cleaned_data_df['end'] <= (x+ asr_delta)) & (cleaned_data_df['start'] >= x)].index.tolist()   )
    
    filtered_df = filtered_df[filtered_df['asrs'].apply(lambda x: len(x) > 1)]

    # drop column with empty lists
    filtered_df['asrs_word_count'] = filtered_df['asrs'].apply(lambda indices: len(
    set(cleaned_data_df.loc[[i for i in indices if i != -1], 'text'].str.split().sum())))
    
    # filter df with asrs_words <10
    filtered_df = filtered_df[filtered_df['asrs_word_count'] > word_threshold]
    filtered_df['asrs_txt'] = filtered_df['asrs'].apply(lambda indices: ', '.join( [cleaned_data_df.loc[i, 'text'] if i != -1 else '<event>' for i in indices]))
    # remove columns  position   team visibility  time_diff_before  time_diff_after
    filtered_df.drop(columns=['position', 'team', 'visibility',
                     'time_diff_before', 'time_diff_after'], inplace=True)
    # add a column with gameid
    filtered_df['gameid'] = row['Game/Half']
    all_df.append(filtered_df)

    # print(len(filtered_df))
    # breakpoint()

all_df = pd.concat(all_df, ignore_index=True)
print(all_df.head())
print(all_df.describe())
print(all_df.label.value_counts())
all_df['label'] = all_df['label'].apply(lambda x: event_to_group.get(x), None)
all_df = all_df[all_df['label'].notna()]
all_df['id'] = all_df.index
breakpoint()
# save as csv
all_df.to_csv("SN-echoes-class-v0.csv", index=False)
all_df['script'] = all_df.apply( lambda x: f"ffmpeg -n -i './SoccerNet_224p/{x.gameid}_224p.mkv' -c:v libx264 -c:a aac -ss {x.gameTime-8} -to {x.gameTime+8} './SN-echoes-class-v0/{x.id}.mp4'", axis=1)
all_df['script'].to_csv("SN-echoes-class-v0.txt", index=False, header=False)
os._exit(0)
