import torch
import torch.nn as nn

taille_vocabulaire = 100 
d_model = 4 

couche_sortie = nn.Linear(d_model, taille_vocabulaire)


vecteur_contexte = torch.rand(1, d_model) 


predictions = couche_sortie(vecteur_contexte)
print("Predictions sur 100 mots :", predictions)


mot_attendu = torch.tensor([42])


fonction_de_perte = nn.CrossEntropyLoss()

perte = fonction_de_perte(predictions, mot_attendu)

print(f"Prédictions (les 5 premiers scores sur 100) : {predictions[0][:5]}")
print(f"\nLa note de pénalité (Loss) est de : {perte.item():.4f}")