from __future__ import annotations

import torch
from torch import nn

from utils.embeddings import sinusoidal_embedding


class FlowMatchingMLP(nn.Module):
	def __init__(
		self,
		data_dim: int,
		hidden_dim: int = 256,
		time_embedding_dim: int = 128,
		num_hidden_layers: int = 5,
	):
		super().__init__()
		self.data_dim = data_dim
		self.hidden_dim = hidden_dim
		self.time_embedding_dim = time_embedding_dim
		self.num_hidden_layers = num_hidden_layers

		layers: list[nn.Module] = []
		input_dim = data_dim + time_embedding_dim
		layers.append(nn.Linear(input_dim, hidden_dim))
		layers.append(nn.ReLU())
		for _ in range(num_hidden_layers - 1):
			layers.append(nn.Linear(hidden_dim, hidden_dim))
			layers.append(nn.ReLU())
		layers.append(nn.Linear(hidden_dim, data_dim))
		self.network = nn.Sequential(*layers)

	def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		if t.ndim == 0:
			t = t.unsqueeze(0).expand(z.shape[0])
		elif t.ndim == 1 and t.shape[0] == 1 and z.shape[0] > 1:
			t = t.expand(z.shape[0])
		elif t.ndim > 1:
			t = t.reshape(z.shape[0])
		t_emb = sinusoidal_embedding(t.to(z.device), self.time_embedding_dim)
		if t_emb.shape[0] != z.shape[0]:
			t_emb = t_emb.expand(z.shape[0], -1)
		inputs = torch.cat([z, t_emb], dim=-1)
		return self.network(inputs)
