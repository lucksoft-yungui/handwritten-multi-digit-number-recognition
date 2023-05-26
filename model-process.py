import torch
from handwritten_multi_digit_number_recognition.lit_models import CTCLitModel

# 加载 PyTorch Lightning 模型
pl_model = CTCLitModel.load_from_checkpoint('lit_model.ckpt')

# 获取模型的权重
model_weights = pl_model.state_dict()

# 保存模型权重
torch.save(model_weights, "model_weights.pth")

# 初始化你的模型
model = CTCLitModel(...)  # 在这里填入你的模型参数

# 加载权重
model.load_state_dict(torch.load("model_weights.pth"))

# 切换到评估模式
model.eval()
