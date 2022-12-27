import torch
import torch.nn as nn


class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


class Unflatten(nn.Module):
	def __init__(self, channel, height, width):
		super(Unflatten, self).__init__()
		self.channel = channel
		self.height = height
		self.width = width

	def forward(self, input):
		return input.view(input.size(0), self.channel, self.height, self.width)


class ConvVAE(nn.Module):

	def __init__(self, input_channels=1, z_dim=20):
		super(ConvVAE, self).__init__()

		self.z_dim = z_dim

		self.encoder = nn.Sequential(
			nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),  # [?,128,7,7]
			Flatten(),
			nn.Linear(6272, 1024),
			nn.ReLU()
		)

		# hidden => mu
		self.fc1 = nn.Linear(1024, self.z_dim)

		# hidden => logvar
		self.fc2 = nn.Linear(1024, self.z_dim)

		self.decoder = nn.Sequential(
			nn.Linear(self.z_dim, 1024),
			nn.ReLU(),
			nn.Linear(1024, 6272),
			nn.ReLU(),
			Unflatten(128, 7, 7),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
			nn.Sigmoid()
		)

	def encode(self, x):
		h = self.encoder(x)
		mu, logvar = self.fc1(h), self.fc2(h)
		return mu, logvar	# 直接返回方差的对数

	def decode(self, z):
		z = self.decoder(z)
		return z

	def reparameterize(self, mu, logvar):
		# 从 Normal(μ, σ^2)中采样一个Z
		# 相当于从Normal(0, 1)采样一个ε
		# Z = μ + ε × σ
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar
