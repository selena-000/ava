import os

audio_dir = "merg_data/train/audio"
video_dir = "merg_data/train/video"

# 获取不带后缀的文件名集合
audio_files = {os.path.splitext(f)[0] for f in os.listdir(audio_dir) if f.endswith(".wav")}
video_files = {os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith(".mp4")}

# 两边都存在的交集
common_files = audio_files & video_files

# 要删除的文件（只在一边有）
audio_only = audio_files - common_files
video_only = video_files - common_files

# 删除 audio 中多余的 .wav
for fname in audio_only:
    fpath = os.path.join(audio_dir, f"{fname}.wav")
    if os.path.exists(fpath):
        os.remove(fpath)
        print(f" Deleted audio: {fpath}")

# 删除 video 中多余的 .mp4
for fname in video_only:
    fpath = os.path.join(video_dir, f"{fname}.mp4")
    if os.path.exists(fpath):
        os.remove(fpath)
        print(f" Deleted video: {fpath}")
