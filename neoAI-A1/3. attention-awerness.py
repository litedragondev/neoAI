import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 4
seq_len = 3

entrees_mots = torch.rand(seq_len, d_model) 
print("Mots entrés :", entrees_mots)

W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)
print("Wq = ", W_q, "; Wk = ", W_k, "; Wv = ", W_v)

Q = W_q(entrees_mots) 
K = W_k(entrees_mots) 
V = W_v(entrees_mots) 
print("Q = ", Q, "; K = ", K, "; V = ", V)

scores_bruts = torch.matmul(Q, K.transpose(0, 1))
print("Scores bruts de base :", scores_bruts)

scores_bruts = scores_bruts / (d_model ** 0.5)
print("Scores bruts stabilisés :", scores_bruts)

poids_attention = F.softmax(scores_bruts, dim=-1)
print("Poids Softmax :", poids_attention)

vecteurs_finaux = torch.matmul(poids_attention, V)

print("Vecteurs enrichis par le contexte :")
print(vecteurs_finaux)