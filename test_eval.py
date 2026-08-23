import sys
sys.path.append("/home/ghostblaster08/Projects/speech_synth/speech-synthesis")
from run_ablation_datasets import train_and_eval
print(train_and_eval("extracted_features/ablation_Pure_DSP.pt"))
