https://github.com/ollama/ollama

-- 
https://github.com/ggerganov/llama.cpp/blob/952d03dbead16e4dbdd1d3458486340673cc2465/convert-lora-to-ggml.py#L51
https://github.com/ggerganov/llama.cpp/blob/ee52225067622babc277371511b8124884e1c797/gguf-py/gguf/constants.py#L187
python llm/llama.cpp/convert-lora-to-ggml.py model phi3

git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf 

Modelfile:

FROM ./Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf
ADAPTER ./model/ggml-adapter-model.bin
TEMPLATE """<s>{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>"""
PARAMETER stop <|endoftext|>
PARAMETER stop <|assistant|>
PARAMETER stop <|end|>
PARAMETER num_ctx 4096

ollama create mario2 -f Modelfile

ollama run mario2

ollama rm mario2

