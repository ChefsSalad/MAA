import torch

if torch.cuda.is_available():
    print("✅ GPU 可用：", torch.cuda.get_device_name(0))
else:
    print("❌ 当前环境无 GPU，可用 CPU")
