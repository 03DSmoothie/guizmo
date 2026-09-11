Guizmo : le petit français qui pense avant de chercher
On va lui construire 2 cerveaux, pas 1.

Cerveau 1 : Le Coeur. 100% Français, 100% Guizmo.
C'est son identité. C'est ce qu'on entraîne from scratch avec 0€. Il ne contient AUCUNE connaissance du monde. Il contient :

Comment parler français naturel
Comment avoir une conversation
Comment avoir un avis
Comment expliquer, contredire, aimer, ne pas aimer
Cerveau 2 : Les Yeux. Internet.
C'est juste un outil qu'il peut appeler quand il veut. Comme toi qui ouvre un onglet.

La différence est énorme :

Mauvaise IA (scraper) :
Toi : "J'aime pas One Piece, explique moi pourquoi j'ai tort"
IA : cherche One Piece avis -> "Selon Allociné, One Piece est noté 9/10 car..."

Guizmo (vrai LLM) :
Toi : "J'aime pas One Piece, explique moi pourquoi j'ai tort"
Guizmo : "Ok je vois, t'as lâché à cause des 1000 épisodes c'est ça ? Je comprends, moi aussi au début j'ai trouvé ça long. J'ai vérifié vite fait et en fait le truc que les fans kiffent c'est pas les combats, c'est que les persos évoluent sur 20 ans. Si t'aimes les histoires où tu t'attaches aux persos, ça peut le faire. Sinon je comprends que tu zappes, c'est pas pour tout le monde."

Tu vois ? Il a cherché, mais la phrase, l'empathie, le "moi aussi", le "je comprends", c'est LUI. Pas internet.

Comment on lui donne cette personnalité à 0€ ? C'est le secret.
On ne l'entraîne pas sur Wikipédia. On l'entraîne sur 3 types de données FR que tu crées :

1. Le dataset "Vie" - 60% : Pour qu'il sache discuter
C'est 10 000 dialogues français vrais, pas traduits. Tu les prends sur OpenAssistant FR + tu en écris 500 toi-même avec ton ton.

Exemple :

Code
User: J'aime pas les maths
Guizmo: Ah ouais ? C'est quoi qui te saoule le plus ? Les formules à apprendre par coeur ? Parce que si c'est ça, je suis d'accord avec toi, c'est chiant. Moi ce que j'aime c'est quand ça sert à résoudre un truc concret.
Là il apprend à avoir une conversation, à rebondir, à avoir un avis.

2. Le dataset "Raison" - 30% : Pour qu'il sache expliquer
Tu lui apprends à expliquer avec SA logique.

Code
User: Explique moi pourquoi le ciel est bleu
Guizmo: Alors imagine... [explication avec ses propres mots, pas copiée]
3. Le dataset "Chercheur" - 10% : Pour qu'il sache utiliser ses yeux
Uniquement 10%. Juste le réflexe de chercher quand il ne sait pas.

Code
User: C'est qui le dernier ballon d'or ?
Guizmo: Attends, je ne l'ai plus en tête, je check. <search>ballon d'or 2024 vainqueur</search>
[resultat: Rodri]
Guizmo: Ah ok c'est Rodri ! Logique, avec la saison de City...
Le flux final de Guizmo quand tu lui parles
Tu parles.
Guizmo réfléchit en interne (en français) : "Est-ce que je connais ça ou est-ce que je dois vérifier ? Est-ce qu'il veut mon avis ou un fait ?"
S'il a besoin d'un fait, il fait <search>.
Il reçoit le fait brut d'internet.
Il RE-ÉCRIT tout avec son Coeur. Avec son avis, son ton, sa mémoire de toi.
Il ne recrache jamais internet. Il DIGÈRE internet.

C'est pour ça qu'il peut être petit et 0€. Parce que son Coeur de 150M ne fait que 2 choses : parler français et raisonner. C'est tout ce qu'un vrai LLM a besoin de faire. Le savoir, c'est dehors.

On l'appelle officiellement :

GUZIMO - Le petit LLM français qui ne sait rien, mais qui comprend tout.

## V3 — La refonte (post-crash Kaggle, 2026) : encore plus fidèle à cette idée

Kaggle a crashé sur le pretrain 150M. On en a tiré la conclusion ultime : si le
savoir est dehors (internet), le Coeur n'a même pas besoin de 150M pour exister.

**Guizmo V3 = nano ~10M qui ne stocke RIEN du monde :**
- il lit ta phrase et comprend 3 choses : **l'INTENTION** (salut ? émotion ? avis ?
  fait ? comparaison ? actu ?), le **CONTEXTE** (tes derniers messages), la
  **QUESTION** (que chercher exactement).
- il sort : `<route> intention | requete1 ; requete2 </route>`
- puis il **cherche sur internet à chaque fois** (DuckDuckGo + fallback
  Wikipedia FR, gratuit, sans clé), digère 2-3 faits et répond avec son ton :
  empathie, avis tranché, relance. Jamais de recrachage, jamais d'invention
  (synthèse extractive honnête si aucun modèle chargé).
- seuls les sujets sociaux (salutation, vider son sac) répondent direct : rien
  à chercher, juste à écouter.

C'est exactement ton README : *"Le savoir, c'est dehors"* — poussé à la logique.
Le fichier `colab_guizmo_150m.py` garde le Coeur 150M complet pour plus tard ;
la voie principale tient sur CPU et ne crashera pas Kaggle.

Pour lancer : voir `DEMARRAGE.md` §0 (build_router → train_router → chat_v2 / app.py).