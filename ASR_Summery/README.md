# 会议音频转录+总结项目
## 音频转录（ASR）
使用来自[FunASR](https://github.com/modelscope/FunASR/blob/main/README_zh.md)的开源模型，进行中文语音识别。
### run
运行```ASR_stream.py```，使用预训练模型进行**实时**流式语音识别。文件路径通过```wav_file```设置。
运行```ASR_nonStream.py```，使用预训练模型进行**非实时**语音识别。文件路径通过```input```设置。

## 会议内容整理和总结（Summery）