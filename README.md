![neoAI hero image](https://github.com/litedragondev/neoAI/blob/main/files/hero.png?raw=true)

This repository is the home of my learnings about Artificial Intelligence and Neuronal Networks. 
This code is free of use and of reading. All of the code is written in Python with variable and function names in French.
Thanks for the attention you gave to my work!

*Ce repo contient une trace de mes apprentissages sur l'Intelligence Artificielle et les Réseaux Neuronaux.
Ce code est libre d'usage et de lecture.
Tout le code est écrit en Python avec les noms de variables et de fonctions en Français.
Merci de l'attention que vous portez à mon travail !*

## My first ever model : neoAI-A1

This AI model is **not** a LLM or any other type of modern AI model. For now, it is only a Predictive Model.

It is honestly pretty **irresponsive and uncomprehensive** for now. With this base I will try to create a more comprehensive version in the future.

All files are presents under ```neoAI-A1``` folder. Here is a list of all of them :
* ```1. text-processing.py``` changes words into tokens
* ```2. neuron-calculation.py``` shows if a neuron will be on or not given simulated tokens and weights
* ```3. attention-awerness.py``` adds context awerness with context-enriched vectors to guide the model to the proper sense of a word
* ```4. positional-encoding.py``` gives an Index to each token to remake the sentence by reordering each tokens by those indexes
* ```5. loss-calculation.py``` for reinforcement learning, inflicts a penalty if the result is not the one expected
* ```6. machine-learning.py``` uses the loss calculation to train the model with reinforcement learning
* ```7. inference.py``` using this, the model can now generate a "sentence" by itself
* ```neoaiPredictiveA1.py``` is the assembly of all functions I learned to use the Predictive model into one single Python Script. With this script, the model self trains and then regenarate itself the text given at the start
* ```neoAIPredictiveA1-interactive.py```  this version lets the user interact with the model. It's essentially the same as ```neoaiPredictiveA1.py``` but with user interaction **(VERY BUGGY AND NOT ACCURATE AT ALL)**