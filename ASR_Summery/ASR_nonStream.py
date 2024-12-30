
from funasr import AutoModel
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
model = AutoModel(model="paraformer-zh",  vad_model="fsmn-vad", punc_model="ct-punc", disable_update=True
                  # spk_model="cam++"
                  )
res = model.generate(input="eg.wav", 
            batch_size_s=300, 
            hotword='带电'
            )
print(res[0]['text'])
with open('eg.txt','w') as f:
    f.write(res[0]['text'])