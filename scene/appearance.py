import torch.nn as nn

class AppearanceMLP(nn.Module):
    def __init__(self, sh_dim=16, app_dim=32, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(app_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, sh_dim)  # 输出和SH系数
        self.act = nn.ReLU()

    def forward(self, app_code):
        x = self.act(self.fc1(app_code))
        # out = torch.sigmoid(self.fc2(x))  # 映射到 [0,1]
        out = self.fc2(x)
        sh_res_mlp = out.view(3, -1)
        return sh_res_mlp


class AppearanceEmbedding(nn.Module):
    def __init__(self, num_images, app_dim=32):
        super().__init__()
        # 每张图像一个可学习的 appearance 向量
        self.embeddings = nn.Embedding(num_images, 3*app_dim)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)

    def forward(self, image_ids):
        return self.embeddings(image_ids)  # [N, 3*app_dim]
