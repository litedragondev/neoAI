import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neoaiPredictiveA1 import NeoAI_A1

dialogues_base = [
    ("Bonjour !", "Salut, comment puis-je t'aider ?"),
    ("Salut", "Salut, comment puis-je t'aider ?"),
    ("Coucou", "Salut, comment puis-je t'aider ?"),
    ("Qui es-tu ?", "Je suis une IA créée en Python."),
    ("Que manges-tu ?", "Je me nourris exclusivement de tenseurs.")
]

def ajouter_fautes(texte, probabilite=0.08):
    texte_abime = list(texte)
    for i in range(len(texte_abime)):
        if random.random() < probabilite and texte_abime[i].isalpha():
            texte_abime[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(texte_abime)

dialogues_entrainement = []
for user_msg, ia_msg in dialogues_base:
    dialogues_entrainement.append(f"<USER> {user_msg} <IA> {ia_msg} <FIN>")
    user_bruite = ajouter_fautes(user_msg, probabilite=0.1)
    dialogues_entrainement.append(f"<USER> {user_bruite} <IA> {ia_msg} <FIN>")

alphabet_robuste = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàâäéèêëîïôöùûüç.,!?'-<>/\n "
texte_complet = "".join(dialogues_entrainement) + alphabet_robuste
caracteres = sorted(list(set(texte_complet)))
taille_vocabulaire = len(caracteres)

char_to_id = {ch: i for i, ch in enumerate(caracteres)}
id_to_char = {i: ch for i, ch in enumerate(caracteres)}


def obtenir_batch_chat(dialogues, batch_size):
    x_batch = []
    y_batch = []
    
    indices = torch.randint(0, len(dialogues), (batch_size,))
    
    for idx in indices:
        dialogue = dialogues[idx]
        encode = [char_to_id[c] for c in dialogue]
        
        x = torch.tensor(encode[:-1], dtype=torch.long)
        
        y = torch.tensor(encode[1:], dtype=torch.long)
        
        balise = "<IA> "
        index_balise = dialogue.find(balise)
        
        if index_balise != -1:
            index_debut_reponse_dans_y = index_balise + len(balise) - 1
            y[:index_debut_reponse_dans_y] = -100
            
        x_batch.append(x)
        y_batch.append(y)
        
    x_pad = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=0)
    y_pad = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=-100)
    
    return x_pad, y_pad

d_model = 64
seq_len = 128 

modele = NeoAI_A1(taille_vocabulaire, d_model, seq_len)
optimiseur = optim.Adam(modele.parameters(), lr=1e-3)

fonction_perte = nn.CrossEntropyLoss(ignore_index=-100)

print("--- Début du Fine-Tuning robuste ---")
modele.train()

for epoque in range(5000):
    x_batch, y_batch = obtenir_batch_chat(dialogues_entrainement, batch_size=2)
    
    logits = modele(x_batch)
    
    logits_aplatis = logits.view(-1, taille_vocabulaire)
    y_aplati = y_batch.view(-1)
    
    perte = fonction_perte(logits_aplatis, y_aplati)
    
    optimiseur.zero_grad()
    perte.backward()
    optimiseur.step()
    
    if epoque % 500 == 0:
        print(f"Époque {epoque} | Erreur sur les réponses IA : {perte.item():.4f}")


print("\n--- Discussion ---")
modele.eval()

while True:
    question_utilisateur = input("\nVous : ")
    
    if question_utilisateur.lower() in ["quitter", "exit", "stop"]:
        print("Fin de la discussion.")
        break
        
    contexte_depart = f"<USER> {question_utilisateur} <IA> "

    try:
        encode_depart = [char_to_id[c] for c in contexte_depart]
    except KeyError as e:
        print(f"[Erreur] Le caractère {e} n'est pas géré par le dictionnaire.")
        continue

    x_generation = torch.tensor([encode_depart], dtype=torch.long)
    texte_genere = ""
    
    print("IA : ", end="")
    
    for _ in range(100): 
        x_tronque = x_generation[:, -seq_len:]
        
        with torch.no_grad():
            logits = modele(x_tronque)
            
        logits_dernier = logits[:, -1, :] / 0.5 # Température contrôlée
        probabilites = F.softmax(logits_dernier, dim=-1)
        id_choisi = torch.multinomial(probabilites, num_samples=1)
        
        caractere_choisi = id_to_char[id_choisi.item()]
        texte_genere += caractere_choisi
        print(caractere_choisi, end="", flush=True)
        
        x_generation = torch.cat((x_generation, id_choisi), dim=1)
        
        if texte_genere.endswith("<FIN>"):
            break

print("\n--- Conversation terminée ---")