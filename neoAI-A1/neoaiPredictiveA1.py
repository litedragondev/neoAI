import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

texte_brut = """
Le chat mange la souris. Le chien aboie après le facteur.
La souris fuit le chat. Le facteur court vite.
"""

caracteres = sorted(list(set(texte_brut)))
taille_vocabulaire = len(caracteres)

char_to_id = {ch: i for i, ch in enumerate(caracteres)}
id_to_char = {i: ch for i, ch in enumerate(caracteres)}

donnees_encodees = torch.tensor([char_to_id[c] for c in texte_brut], dtype=torch.long)

print(f"Taille du vocabulaire : {taille_vocabulaire} caractères")
print(f"Taille totale du texte : {len(donnees_encodees)} tokens")


class NeoAI_A1(nn.Module):
    def __init__(self, taille_vocabulaire, d_model, seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(taille_vocabulaire, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.couche_sortie = nn.Linear(d_model, taille_vocabulaire)

    def forward(self, x):
        batch_size, longueur_sequence = x.size()
        masque_causal = torch.triu(torch.ones(longueur_sequence, longueur_sequence) * float('-inf'), diagonal=1)
        masque_causal = masque_causal.to(x.device)
        positions = torch.arange(0, longueur_sequence, device=x.device).unsqueeze(0).expand(batch_size, -1)
        h = self.token_embedding(x) + self.position_embedding(positions)
        h_attention, _ = self.attention(query=h, key=h, value=h, attn_mask=masque_causal, is_causal=True)
        h = h + h_attention
        h = h + self.feed_forward(h)
        logits = self.couche_sortie(h)
        return logits

d_model = 128
seq_len = 32
batch_size = 4
learning_rate = 1e-3
epochs = 5000

modele = NeoAI_A1(taille_vocabulaire, d_model, seq_len)
optimiseur = optim.Adam(modele.parameters(), lr=learning_rate)
fonction_perte = nn.CrossEntropyLoss()


def obtenir_batch():
    index_aleatoires = torch.randint(len(donnees_encodees) - seq_len, (batch_size,))
    x = torch.stack([donnees_encodees[i : i + seq_len] for i in index_aleatoires])
    y = torch.stack([donnees_encodees[i + 1 : i + seq_len + 1] for i in index_aleatoires])
    return x, y


print("\n--- Début de l'entraînement ---")
modele.train()

for epoque in range(epochs):
    x_batch, y_batch = obtenir_batch()
    logits = modele(x_batch)
    logits_aplatis = logits.view(-1, taille_vocabulaire)
    y_aplati = y_batch.view(-1)
    perte = fonction_perte(logits_aplatis, y_aplati)
    optimiseur.zero_grad()
    perte.backward()
    optimiseur.step()
    if epoque % 100 == 0:
        print(f"Époque {epoque} | Erreur (Loss) : {perte.item():.4f}")

print("\n--- Génération de texte ---")
modele.eval()
contexte_depart = "Le ch"
x_generation = torch.tensor([[char_to_id[c] for c in contexte_depart]], dtype=torch.long)
caracteres_a_generer = 50
temperature = 0.8

for _ in range(caracteres_a_generer):
    x_generation_tronque = x_generation[:, -seq_len:]
    with torch.no_grad():
        logits = modele(x_generation_tronque)
    logits_dernier_caractere = logits[:, -1, :] / temperature
    probabilites = F.softmax(logits_dernier_caractere, dim=-1)
    id_choisi = torch.multinomial(probabilites, num_samples=1)
    x_generation = torch.cat((x_generation, id_choisi), dim=1)

texte_final = "".join([id_to_char[i.item()] for i in x_generation[0]])
print(f"Résultat : {texte_final}")
