import argparse
import torch
import onnx
from onnx2pytorch import ConvertModel


parser = argparse.ArgumentParser()

parser.add_argument('model', type=str, help='input model path')
parser.add_argument('to', type=str, help='export to .ckpt')

args = parser.parse_args()

# Load onnx model
onnx_model = onnx.load(args.model)

# Convert to pytorch model
pytorch_model = ConvertModel(onnx_model)

# Save as .ckpt
torch.save(pytorch_model.state_dict(), args.to)
