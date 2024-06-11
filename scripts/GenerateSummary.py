import glob
import json
import pandas as pd
import openai
from openai import OpenAI
import os
from tqdm import tqdm

import argparse

parser= argparse.ArgumentParser(description="Generate summary of soccer game")
parser.add_argument("--api_key", type=str, help="OpenAI API key", default="sk-123")
parser.add_argument("--caption",  help="either in caption", action="store_true")
parser.add_argument("--event",  help="either in event", action="store_true")
parser.add_argument("--asr",  help="either in asr", action="store_true")
parser.add_argument("--model", type=str, help="Model name", default="meta-llama/Meta-Llama-3-70B-Instruct")

args= parser.parse_args()
print(args)

client = OpenAI(
    api_key= args.api_key,
    # base_url= "http://g002:8000/v1",
)


assert args.caption or args.event or args.asr , "Atleast one of the options should be selected"

file_name ="Summary1"
if args.caption:
    file_name+= "_caption"
if args.event:
    file_name+= "_event"
if args.asr:
    file_name+= "_asr"

def remove_repeating_entries(json_data):
    cleaned_data = {}
    prev_text = None

    for key, value in json_data.items():
        start, end, text = value
        if text.strip() != prev_text:
            cleaned_data[key] = value
        prev_text = text.strip()

    return cleaned_data


language_df = pd.read_csv("/home/sushant/D1/DataSets/SoccerNet-Echoes/SN-echoes-lang.csv")
language_df = language_df[language_df['Lang'] != "NotAvailable"]

# caption_folder="/global/D1/projects/HOST/Datasets/SoccerNet-Caption/"
caption_folder="/global/D1/projects/HOST/Datasets/SoccerNet-Caption/"
all_files = glob.glob(caption_folder+"**/*.json", recursive=True)
print("Total files: ", len(all_files))

caption_to_event_mapping = {
    'corner': 'Corner',
    'penalty': 'Penalty',
    'r-card': 'Red card',
    'soccer-ball': 'Goal',
    'soccer-ball-own': 'Goal in own post',  # Own goals are usually annotated as goals
    'substitution': 'Substitution',
    'y-card': 'Yellow card',
    'yr-card': 'Yellow->red card'
}



match_set =[]


def formatEvents(row, home, away):
    string_output= f"{row['gameTimex']} Event: {row['label']}"
    if row['team'] !="not applicable":
        team_name = home if row['team'] == "home" else away
        string_output += f" by {team_name}"
    return string_output


def formatCaptions(row):
    return f"{row['gameTimex']} Caption: {row['description']}"

def formatASR(row):
    return f"{row['gameTimex']} Comment: {row['description']}"
all_files.reverse()
# all_files= all_files[125:235]
# all_files= all_files[236:360]

for caption_file in tqdm(all_files):
    # read the json file
    event_file = caption_file.replace("SoccerNet-Caption", "SoccerNet_224p").replace("Labels-caption", "Labels-v2")
    row_1 = language_df[(language_df['Game/Half'] ==  "/".join(caption_file.split("/")[-4:-1])+ "/1")]
    row_2 = language_df[(language_df['Game/Half'] ==  "/".join(caption_file.split("/")[-4:-1])+ "/2")]
    if row_1.empty or row_2.empty:
        continue
    row_1, row_2= row_1.iloc[0], row_2.iloc[0]
    ver_1 = "whisper_"+ (row_1.selected if row_1.Lang=="en" else row_1.selected+"_en")
    ver_2 = "whisper_"+ (row_2.selected if row_2.Lang=="en" else row_2.selected+"_en")

    asr_1_file = caption_file.replace("/global/D1/projects/HOST/Datasets/SoccerNet-Caption/", "/home/sushant/D1/DataSets/SoccerNet-Echoes/Dataset/"+ ver_1+"/").replace("Labels-caption", "1_asr")
    asr_2_file = caption_file.replace("/global/D1/projects/HOST/Datasets/SoccerNet-Caption/", "/home/sushant/D1/DataSets/SoccerNet-Echoes/Dataset/"+ ver_2+"/").replace("Labels-caption", "2_asr")
    try:
        asr_1_df =  pd.DataFrame([[e[0], e[2]] for e in remove_repeating_entries(json.load(open(asr_1_file))['segments']).values()], columns=['gameTime', 'description'])
        asr_2_df =  pd.DataFrame([[e[0], e[2]] for e in remove_repeating_entries(json.load(open(asr_2_file))['segments']).values()], columns=['gameTime', 'description'])
    except FileNotFoundError as e:
        continue
        # breakpoint()
    for id, e in asr_1_df.iterrows():
        # update if in asr_1_df
        asr_1_df.loc[id, 'gameTimex']= f"{int(e['gameTime']//60)}:{int(e['gameTime']%60)}"
        asr_1_df.loc[id, 'half']= 1
    for id, e in asr_2_df.iterrows():
        asr_2_df.loc[id, 'gameTimex']= f"{int(e['gameTime']//60)}:{int(e['gameTime']%60)}"
        asr_2_df.loc[id, 'half']= 2
    #make half and gametime as int
    asr_1_df['gameTime'], asr_1_df['half']= asr_1_df['gameTime'].astype(int), asr_1_df['half'].astype(int)
    asr_2_df['gameTime'], asr_2_df['half']= asr_2_df['gameTime'].astype(int), asr_2_df['half'].astype(int)
    asr_df = pd.concat([asr_1_df, asr_2_df]).sort_values(by=['half', 'gameTime'])

    
    captions_json = json.load(open(caption_file))
    events = json.load(open(event_file))['annotations']
    captions= captions_json['annotations']
    cap_df = pd.DataFrame(captions)
    all_players = {player['hash']: player for team in captions_json['lineup'].values() for player in team['players']}
    game_name_raw = caption_file.split("/")[-2].split(" - ")
    home_team= game_name_raw[1][game_name_raw[1].index(' ') + 1 : game_name_raw[1].rindex(' ')]
    home_team_score = game_name_raw[1].split(" ")[-1]
    away_team =game_name_raw[2].split(" ")[1]
    away_team_score = game_name_raw[2].split(" ")[0]

    
    
    # lineup-->home or away --> players --> facts (type, time, description, linked_player_hash)
    for e in events:
        times = e['gameTime'][4:].split(":")
        e['gameTime'], e['half'] = int(times[0])*60 + int(times[1]), int(e['gameTime'][:2])
        e['gameTimex']= times[0]+":"+times[1]
        
    for c in captions:
        times = c['gameTime'][4:].split(":")
        c['gameTime'], c['half'] = int(times[0])*60 + int(times[1]), int(c['gameTime'][:2])
        c['gameTimex']= times[0]+":"+times[1]
        c['event_label'] = caption_to_event_mapping.get(c['label'], None)
        # remove identified and anonymized keys
        c.pop('identified', None)
        c.pop('anonymized', None)
        c.pop('position', None)
        c.pop('visibility', None)

   
    events_df = pd.DataFrame(events).sort_values(by=['half', 'gameTime'])
    captions_df = pd.DataFrame(captions).sort_values(by=['half', 'gameTime'])

    ##format data 
    events_df['text']= events_df.apply(lambda x: formatEvents(x,home_team, away_team), axis=1)
    captions_df['text']= captions_df.apply(lambda x: formatCaptions(x), axis=1)
    asr_df['text']= asr_df.apply(lambda x: formatASR(x), axis=1)


    # concat events_df and captions_df
    # empty df
    text_df= pd.DataFrame()
    if args.caption:
        text_df = pd.concat([captions_df, text_df])
    if args.event:
        text_df = pd.concat([events_df, text_df])
    if args.asr:
        text_df = pd.concat([asr_df, text_df])

    text_df = text_df.sort_values(by=['half', 'gameTime'])  
    # text_df = pd.concat([events_df, captions_df, asr_df]).sort_values(by=['half', 'gameTime'])
    # text_df.drop(columns=['visibility', 'description', 'label', 'important', 'event_label','team'], inplace=True)
    first_half= text_df[text_df['half'] == 1]['text']
    second_half= text_df[text_df['half'] == 2]['text']
    first_half_text, second_half_text = "\n".join(first_half), "\n".join(second_half)

    post_instruction= '''
    #######TASK 
    Provide a factually correct and detailed textual description of the game compiled from the information presented above from different sources, in more than 1000 words.'''
    if args.caption:
        post_instruction+= '''
            "Cap:": represents the captions published in the online portal usually within 30 seconds of the event.'''
    if args.event:
        post_instruction+= '''
            "E:" represents events recorded exactly after the event happened.'''
    if args.asr:
        post_instruction+= '''
            "Com.": represents the real-time noisy commentary from the commentator as the game progresses.'''
    post_instruction+= '''
    Don't mention sources of information like commentators or captions, write as if you saw them live.
    '''

    message= [{"role": "system", "content": '''
    You are an intelligent soccer game describer for BBC news. You will be given temporal event information from two game halves.
    You will return a very long and factually correct detailed description of the game in plain text in more than 1000 words without missing any information provided.
    '''
    },{"role": "user", "content": f'''
    {home_team} {home_team_score} - {away_team_score} {away_team}\n\nFirst Half:\n\n{first_half_text}\n\nSecond Half:\n{second_half_text}
    \n{post_instruction}
    '''
    },]
    dump_file_name = caption_file.replace("/global/D1/projects/HOST/Datasets/SoccerNet-Caption/",  "/home/sushant/D1/DataSets/SoccerNet-Echoes/Dataset_summary/").replace("Labels-caption", file_name)

    # check if file already exists
    if os.path.exists(dump_file_name):
        print(f"File already exists: {dump_file_name}")
        continue
    # make dirs
    os.makedirs(os.path.dirname(dump_file_name), exist_ok=True)
    try:
        chat_response = client.chat.completions.create(model=args.model, messages=message)
        text= chat_response.choices[0].message.content

    except openai.BadRequestError as e:
        print(e)
        print("Error in processing:", caption_file)
        breakpoint()
        continue
    with open(dump_file_name, "a") as f:
        data={
            "input": message[1]['content'],
            "output": text
        }
        json.dump(data, f, indent=4)
    print(f"Dumped: {dump_file_name}")
    
    