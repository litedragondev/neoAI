import torch
import torch.nn.functional as F

entree = torch.tensor([0.5, -1.2, 0.8])

poids = torch.tensor([1.5, -0.5, 2.0])

biais = torch.tensor([-0.2])

score_brut = torch.dot(entree, poids) + biais

print(f"Score brut avant activation : {score_brut.item():.4f}")
# (0.5 * 1.5) + (-1.2 * -0.5) + (0.8 * 2.0) - 0.2 
# = 0.75 + 0.6 + 1.6 - 0.2 = 2.75

score_final = F.relu(score_brut)

print(f"Score final (signal transmis) : {score_final.item():.4f}")