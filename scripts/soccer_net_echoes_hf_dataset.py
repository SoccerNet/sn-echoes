# Copyright 2024 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ASR Dataset for various football leagues and seasons"""

import json
import os

import datasets


_CITATION = """\
@article{Gautam2024May,
	author = {Gautam, Sushant and Sarkhoosh, Mehdi Houshmand and Held, Jan and Midoglu, Cise and Cioppa, Anthony and Giancola, Silvio and Thambawita, Vajira and Riegler, Michael A. and Halvorsen, P{\aa}l and Shah, Mubarak},
	title = {{SoccerNet-Echoes: A Soccer Game Audio Commentary Dataset}},
	journal = {arXiv},
	year = {2024},
	month = may,
	eprint = {2405.07354},
	doi = {10.48550/arXiv.2405.07354}
}
"""

_DESCRIPTION = """\
This dataset contains Automatic Speech Recognition (ASR) data for various football leagues and seasons in SoccerNet. 
The dataset includes ASR outputs from Whisper v1, v2, and v3, along with their English-translated versions.
"""

_HOMEPAGE = "https://github.com/SoccerNet/sn-echoes"

_LICENSE = "cc-by-4.0"

_URLS = {
    "whisper_v1": "whisper_v1/",
    "whisper_v1_en":  "whisper_v1_en/",
    "whisper_v2":  "whisper_v2/",
    "whisper_v2_en":  "whisper_v2_en/",
    "whisper_v3":  "whisper_v3/",
    "whisper_v3_en":  "whisper_v3_en/",
}


class FootballASRDataset(datasets.GeneratorBasedBuilder):
    """ASR Dataset for various football leagues and seasons"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="whisper_v1", version=VERSION, description="Contains ASR from Whisper v1"),
        datasets.BuilderConfig(name="whisper_v1_en", version=VERSION,
                               description="English-translated datasets from Whisper v1"),
        datasets.BuilderConfig(
            name="whisper_v2", version=VERSION, description="Contains ASR from Whisper v2"),
        datasets.BuilderConfig(name="whisper_v2_en", version=VERSION,
                               description="English-translated datasets from Whisper v2"),
        datasets.BuilderConfig(
            name="whisper_v3", version=VERSION, description="Contains ASR from Whisper v3"),
        datasets.BuilderConfig(name="whisper_v3_en", version=VERSION,
                                description="English-translated datasets from Whisper v3"),
    ]

    DEFAULT_CONFIG_NAME = "whisper_v1"

    def _info(self):
        features = datasets.Features(
            {
                "segment_index": datasets.Value("int32"),
                "start_time": datasets.Value("float"),
                "end_time": datasets.Value("float"),
                "transcribed_text": datasets.Value("string"),
                "game": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(
            "https://codeload.github.com/SoccerNet/sn-echoes/zip/refs/heads/main") + "/sn-echoes-main/Dataset/"
        print("data_dir", {"data_dir": os.path.join(data_dir + urls), })
        version_name = urls.replace("/", "").replace("_", ".")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": os.path.join(data_dir + urls),
                },)
        ]

    def _generate_examples(self, data_dir,):
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(root, file), encoding="utf-8") as f:
                        data = json.load(f)
                        for segment_index, segment_data in data["segments"].items():
                            filename = "/".join(root.split("/")
                                                [-3:])+"/"+str(file[0])
                            yield f"{filename}_{segment_index}", {
                                "segment_index": segment_index,
                                "start_time": segment_data[0],
                                "end_time": segment_data[1],
                                "transcribed_text": segment_data[2],
                                "game": filename,
                            }


# RUN: datasets-cli test  scripts/soccer_net_echoes_hf_dataset.py --save_info --all_configs
# make sure latest data is downloaded form github, else set force_download set to True
# the dataset will be prepared at ~/.cache/huggingface/datasets/soccer_net_echoes_hf_dataset/
# remove all *.lock files in the directory:
# RUN: rm ~/.cache/huggingface/datasets/soccer_net_echoes_hf_dataset/**/*.lock
# RUN: cd ~/.cache/huggingface/datasets/soccer_net_echoes_hf_dataset/
# HERE are the new files
# Upload to HF datasets (will create a new commit and DOI)
# Bring and edit readme from https://huggingface.co/datasets/SoccerNet/SN-echoes/tree/main (to prevent double commit)
# CAREFUL  CAREFUL CAREFUL, works for the first time only on empty repo
# huggingface-cli upload SoccerNet/SN-echoes  . .  --repo-type dataset