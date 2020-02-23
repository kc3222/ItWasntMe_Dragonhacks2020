# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './fast-style-transfer-master')
sys.path.insert(2, './fast-style-transfer-master/src')
import os

from evaluate import ffwd_to_img

ffwd_to_img('voila/sharingan dog.png', 'portrait.jpg', 'wave.ckpt')