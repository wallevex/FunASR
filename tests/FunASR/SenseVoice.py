from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

sense_voice_small = "iic/SenseVoiceSmall"

# 首先会将model当做本地路径从本地缓存找模型，如果找不到会默认从modelscope下载模型
model = AutoModel(
    model=sense_voice_small, # asr模型路径
    model_version="master", # 如果不配置model_version，会默认为master
    # vad_model="fsmn-vad",
    # vad_model_revision="v2.0.4",
    # vad_kwargs={"max_single_segment_time": 30000},
    # punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    # punc_model_revision="v2.0.4",
    # punc_kwargs={},
    device="cuda:0", # 使用cuda
    disable_update=True #不更新funasr版本
)

res = model.generate(
    input="../audios/乐不可知.wav",
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