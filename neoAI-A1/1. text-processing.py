import torch
import torch.nn as nn

texte_entrainement = "le chat mange la souris et le chien aboie"
mots = texte_entrainement.split()

vocabulaire = list(set(mots))
print("Vocabulaire :", vocabulaire)

word_to_id = {mot: i for i, mot in enumerate(vocabulaire)}
id_to_word = {i: mot for i, mot in enumerate(vocabulaire)}

print("Dictionnaire :", word_to_id)

phrase_a_traduire = ["le", "chat", "mange"]
tokens = [word_to_id[mot] for mot in phrase_a_traduire]

print("Phrase en tokens :", tokens)

taille_vocabulaire = len(vocabulaire)
taille_vecteur = 3

embedding_layer = nn.Embedding(num_embeddings=taille_vocabulaire, embedding_dim=taille_vecteur)

tokens_tensor = torch.tensor(tokens)

vecteurs = embedding_layer(tokens_tensor)

print("\nVecteurs finaux :")
print(vecteurs)