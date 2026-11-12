import torch
import torch.nn.functional as F

# Dictionnaire simulé
id_to_word = {0: "pomme", 1: "dort", 2: "le", 3: "mange", 4: "chat"}

# Logit simulés
logits_bruts = torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0]) 

temperature = 0.8
mots_a_generer = 5
phrase_generee = []

for i in range(mots_a_generer):

    logits_ajustes = logits_bruts / temperature

    probabilites = F.softmax(logits_ajustes, dim=-1)
    
    id_choisi = torch.multinomial(probabilites, num_samples=1).item()
    
    mot_choisi = id_to_word[id_choisi]
    phrase_generee.append(mot_choisi)
    

print("Phrase générée :", " ".join(phrase_generee))