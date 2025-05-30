from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

paraformer_vad_punc = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
paraformer = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
paraformer_hotword = "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"

# 首先会将model当做本地路径从本地缓存找模型，如果找不到会默认从modelscope下载模型
model = AutoModel(
    model=paraformer_vad_punc, # asr模型路径
    model_version="master", # 如果不配置model_version，会默认为master
    vad_model="fsmn-vad",
    vad_model_revision="master",
    vad_kwargs={"max_single_segment_time": 30000},
    punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    punc_model_revision="master",
    punc_kwargs={},
    device="cuda:0", # 使用cuda
    disable_update=True #不更新funasr版本
)

res = model.generate(
    input="audios/长句子.wav",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)

print(res[0]["text"])

text = rich_transcription_postprocess(res[0]["text"])
print(text)