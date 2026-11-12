import torch
import math

seq_len = 3
d_model = 4

encodage_position = torch.zeros(seq_len, d_model)

positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

diviseur = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

encodage_position[:, 0::2] = torch.sin(positions * diviseur)

encodage_position[:, 1::2] = torch.cos(positions * diviseur)

print("Matrice d'Encodage Positionnel (les tatouages) :")
print(encodage_position)

vecteurs_mots = torch.rand(seq_len, d_model)

vecteurs_prets_pour_attention = vecteurs_mots + encodage_position