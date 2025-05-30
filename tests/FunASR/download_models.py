#SDK模型下载
from modelscope import snapshot_download

snapshot_download('iic/SenseVoiceSmall') # SenseVoice多语言语音理解模型Small
snapshot_download('iic/SenseVoiceSmall-onnx') # SenseVoice多语言语音理解模型Small-onnx
snapshot_download('iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch') # Paraformer语音识别-中文-通用-16k-离线-large-长音频版（集成了vad、punc的paraformer，是时间戳模型）
snapshot_download('iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch') # Paraformer语音识别-中文-通用-16k-离线-large-pytorch（paraformer标准版）
snapshot_download('iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404') # Paraformer语音识别-中文-通用-16k-离线-large-热词版（paraformer热词版）
snapshot_download("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online") # Paraformer语音识别-中文-通用-16k-实时-large（paraformer在线版）

snapshot_download("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch") # FSMN语音端点检测-中文-通用-16k
snapshot_download("iic/speech_fsmn_vad_zh-cn-16k-common-onnx") # FSMN语音端点检测-中文-通用-16k-onnx

snapshot_download("iic/punc_ct-transformer_cn-en-common-vocab471067-large") # CT-Transformer标点-中英文-通用-large
snapshot_download("iic/punc_ct-transformer_cn-en-common-vocab471067-large-onnx") # CT-Transformer标点-中英文-通用-large-onnx
snapshot_download("iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727") # CT-Transformer标点-中文-通用-实时

snapshot_download('iic/speech_ngram_lm_zh-cn-ai-wesp-fst') # Ngram语言模型-中文-ai-wesp-fst
snapshot_download('thuduj12/fst_itn_zh') # 基于FST的中文ITN
