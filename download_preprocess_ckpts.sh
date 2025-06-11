mkdir -p checkpoints
cd checkpoints
mkdir -p humanparsing
cd humanparsing
wget -O parsing_atr.onnx https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_atr.onnx
wget -O parsing_lip.onnx https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_lip.onnx
