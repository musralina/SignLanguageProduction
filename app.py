import numpy as np
import os
import math
import time
import mediapipe as mp
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from huggingface_hub import notebook_login
import pandas as pd
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import csv
import warnings
warnings.filterwarnings("ignore")


model_name = "musralina/helsinki-opus-de-en-fine-tuned-wmt16-finetuned-src-to-trg"
# notebook_login()


model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f'Pre-trained model: {model_name} is downloaded from the Huggingface platform')

