import sys
import subprocess

if len(sys.argv) < 2:
    sys.exit(1)

PAPER_DIR = sys.argv[1]

steps = [
    ("MergeAudio.py", "合并音频"),
    ("GenSub.py", "生成字幕"),
    ("GenMovie.py", "生成视频")
]

for script, desc in steps:
    print(f"\n==== {desc} ====")
    ret = subprocess.call([sys.executable, script, PAPER_DIR])
    if ret != 0:
        print(f"{desc} Pipeline Failed")
        sys.exit(ret)
