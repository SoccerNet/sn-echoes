# SoccerNet-Echoes
Official repo for the paper: [SoccerNet-Echoes: A Soccer Game Audio Commentary Dataset](https://arxiv.org/abs/2405.07354).

## Dataset 
Each folder inside the **Dataset** directory is categorized by league, season, and game. Within these folders, JSON files contain the transcribed and translated game commentary.

```python


📂 Dataset
├── 📁 whisper_v1
│   ├── 🏆 england_epl
│   │   ├── 📅 2014-2015
│   │   │   └── ⚽ 2016-03-02 - 23-00 Liverpool 3 - 0 Manchester City
│   │   │       ├── ☁️ 1_asr.json
│   │   │       └── ☁️ 2_asr.json
│   │   ├── 📅 2015-2016
│   │   └── ...
│   ├── 🏆 europe_uefa-champions-league
│   └── ...
├── 📁 whisper_v1_en
│   └── ...
├── 📁 whisper_v2
│   └── ...
├── 📁 whisper_v2_en
│   └── ...
├── 📁 whisper_v3
│   └── ...

whisper_v1: Contains ASR from Whisper v1.
whisper_v1_en: English-translated datasets from Whisper v1.
whisper_v2:  Contains ASR from Whisper v2.
whisper_v2_en:  English-translated datasets from Whisper v2.
whisper_v3: Contains ASR from Whisper v3.
```

Each JSON file has the following format:
```python

{
  "segments": {
    segment index (int):[
      start time in second (float),
      end time in second (float),
      transcribed text from ASR
    ]
    ....
  }
}
```
The top-level object is named segments.
It contains an object where each key represents a unique segment index (e.g., "0", "1", "2", etc.).
Each segment index object has the following properties:
```python
start_time: A number representing the starting time of the segment in seconds.
end_time: A number representing the ending time of the segment in seconds.
text: A string containing the textual content of the commentary segment.
```

##Flattened Data on Hugging Face
The hierarchical directory structure with the JSON files has been transformed into a tabular format, consisting of columns for segment index, start and end times, the text (either transcribed or translated), and the game path (represented as a string).
This was divided into three subsets (v1, v2, and v3 of Whisper versions), with each subset further split into "original" (ASR-generated) and "en" (English translated). The flattened representation can be found under https://huggingface.co/datasets/SoccerNet/SN-echoes.

Particular data split and subset can be loaded using the following example code:
```python
from datasets import datasets
dataset = load_dataset("SoccerNet/SN-echoes", 'whisper_v1', split="en")`
```


## Citation
Please cite our work if you use the SoccerNet-Echoes dataset:

<pre><code>
@misc{gautam2024soccernetechoes,
      title={SoccerNet-Echoes: A Soccer Game Audio Commentary Dataset}, 
      author={Sushant Gautam and Mehdi Houshmand Sarkhoosh and Jan Held and Cise Midoglu and Anthony Cioppa and Silvio Giancola and Vajira Thambawita and Michael A. Riegler and Pål Halvorsen and Mubarak Shah},
      year={2024},
      eprint={2405.07354},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      doi={10.48550/arXiv.2405.07354}
}
</code></pre>

