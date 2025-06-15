# -*- coding: utf-8 -*-
"""
Created on Mon May  5 20:14:47 2025

@author: PigFarmer
"""

import whisperx
import os
import json
import torch
import gc
from tqdm import tqdm 

device = "cuda"
batch_size = 16
compute_type = "float16" 


audio_dir = "C:/Users/PigFarmer/Downloads/Private_dataset/private"  
output_txt = "C:/Users/PigFarmer/Downloads/Private_dataset/task1__answer.txt"
output_json = "C:/Users/PigFarmer/Downloads/Private_dataset/task1_answer_timestamps.json"


model = whisperx.load_model("large-v2", device, compute_type=compute_type)

all_results = {}


with open(output_txt, "w", encoding="utf-8") as txt_out:

    audio_files = sorted(
        [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")],
        key=lambda x: int(os.path.splitext(x)[0])  # 提取檔名中的數字部分並排序
    )


    for file_name in tqdm(audio_files, desc="Processing files", unit="file"):
        file_path = os.path.join(audio_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        print(f"🔊 Processing: {file_name}")


        audio = whisperx.load_audio(file_path)
        result = model.transcribe(audio, batch_size=batch_size)


        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        del model_a
        gc.collect()
        torch.cuda.empty_cache()


        transcript = "".join([seg["text"] for seg in result["segments"]]).strip()
        txt_out.write(f"{base_name}\t{transcript}\n")


        segments = []
        for segment in result["segments"]:
            if "words" in segment:
                for word_info in segment["words"]:
                    segments.append({
                        "word": word_info["word"],
                        "start": word_info["start"],
                        "end": word_info["end"]
                    })

        all_results[base_name] = {
            "text": transcript,
            "segments": segments
        }


with open(output_json, "w", encoding="utf-8") as json_out:
    json.dump(all_results, json_out, ensure_ascii=False, indent=None)


del model
gc.collect()
torch.cuda.empty_cache()

print("✅ 完成！已產出 transcripts.txt 與 word_timestamps.json")


