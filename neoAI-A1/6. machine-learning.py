import torch
import torch.nn as nn
import torch.optim as optim

d_model = 4
taille_vocabulaire = 100
couche_sortie = nn.Linear(d_model, taille_vocabulaire)

optimiseur = optim.Adam(couche_sortie.parameters(), lr=0.01)

print("Poids avant l'apprentissage :", couche_sortie.weight[0][:3])

for epoque in range(5):
    vecteur_contexte = torch.rand(1, d_model)
    predictions = couche_sortie(vecteur_contexte)
    mot_attendu = torch.tensor([42])
    
    fonction_de_perte = nn.CrossEntropyLoss()
    perte = fonction_de_perte(predictions, mot_attendu)

    optimiseur.zero_grad()

    perte.backward()

    optimiseur.step()
    
    print(f"Essai n°{epoque + 1} | Note d'erreur (Loss) : {perte.item():.4f}")

print("Poids après l'apprentissage :", couche_sortie.weight[0][:3])