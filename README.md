### Text2Gloss_EncDecFromScratch.ipynb:
This file includes two enc-dec architectures from scratch and applied for NMT task from German text into German Sign Glosses translation. As a dataset PHOENIX-2014-T is applied. (https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)


### Text2Gloss_wmt16_finetuned.ipynb:

This model is a fine-tuned version of mariav/helsinki-opus-de-en-fine-tuned-wmt16 on Phoenix Weather dataset (PHOENIX-2014-T).
The purpose is Neural Machine Translation from German text into German Sign Glosses.
The fine-tuned model can be tested in https://huggingface.co/musralina/helsinki-opus-de-en-fine-tuned-wmt16-finetuned-src-to-trg.


### app.py:
This file includes the code for generation of the Sign Keypoints from the German text. It is generated using the Huggingface fine-tuned model above.
