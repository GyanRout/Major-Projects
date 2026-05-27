import torch.nn as nn
import torch

class RecommenderModel(nn.Module):
    def __init__(self, num_user:int, num_item:int, embedding_dim: int =32, rating_range: tuple = (1.0, 5.0)):
        super().__init__()
        self.user_factor=nn.Embedding(num_user, embedding_dim)
        self.item_factor=nn.Embedding(num_item, embedding_dim)
        self.user_bias=nn.Embedding(num_user,1)
        self.item_bias=nn.Embedding(num_item,1)
        self.y_range=(rating_range[0]-0.1, rating_range[1]+0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_factor.weight, std=0.1)
        nn.init.normal_(self.item_factor.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor)-> torch.Tensor:
        u = self.user_factor(user_ids)
        i = self.item_factor(item_ids)
        u_b = self.user_bias(user_ids)
        i_b = self.item_bias(item_ids)
        dot_product = (u*i).sum(dim=1, keepdim=True)
        x = dot_product + u_b + i_b

        return torch.sigmoid(x)*(self.y_range[1] - self.y_range[0]) + self.y_range[0]