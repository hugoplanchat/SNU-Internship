# Plan de recherche 3 mois — JEPA pour l'ingénierie navale & océanique
**Stage :** SNU Deep Learning Lab (dllab.snu.ac.kr), Dept. of Naval Architecture and Ocean Engineering
**Encadrant :** Prof. Tae-wan Kim (김태완)
**Profil :** 1A CentraleSupélec, full-time (35–40h/semaine, 13 semaines, ~480h)

---

## Cadrage : pourquoi JEPA dans ce labo précisément

Le Pr. Kim a une trajectoire unique : PhD en CAGD avec Gerald Farin (Bézier/B-Spline, kernel Parasolid de Siemens NX), puis bascule vers le deep learning appliqué au maritime depuis ~10 ans. Le labo couvre **trois grandes lignes** dont JEPA peut nourrir chacune :

1. **Perception et navigation autonome (USV/ASV)** — caméras embarquées, SLAM marin, détection d'obstacles. → **V-JEPA 2** est *la* technologie qui correspond : world model vidéo + planning par MPC en latent space, déjà déployé en zero-shot sur robots Franka. L'extension marine est un sujet ouvert.
2. **Génération et analyse de formes (hull forms, propellers, ship CAD)** — surfaces 3D, point clouds. → **Point-JEPA** (WACV 2025) et **3D-JEPA** sont quasiment faits pour ça. C'est aussi là que ton encadrant a une expertise historique forte.
3. **Prédiction d'environnement marin** (vagues, météo, routing) — séries temporelles spatio-temporelles. → JEPA temporel / V-JEPA peuvent être appliqués à des données de mer (ondelettes radar, séquences de wave fields).

**Tu choisiras ton mini-projet en fin de Phase 2 dans une de ces trois lignes**, en concertation avec le Pr. Kim. Le plan est conçu pour que les 8 premières semaines te rendent capable de discuter sérieusement avec lui de la direction à prendre.

---

## Philosophie

JEPA est exigeant : il s'appuie sur 5–6 ans de littérature SSL + Vision Transformers + énergie/probabilité. En 1A tu as les maths, mais pas forcément encore le deep learning à fond. **Donc on ne saute pas les fondations.** Lire un papier JEPA sans avoir intégré DINO, BYOL, MAE = hocher la tête sans rien capter.

**Répartition hebdomadaire indicative :**
- ~50% lecture/théorie (papiers, cours, blogs)
- ~40% code (reproduction, expé)
- ~10% rédaction (notes, journal de bord, blog post)

**Tiens un journal de recherche** (Notion ou Markdown sur GitHub) — un fichier par semaine avec : ce que tu as lu, compris, ce qui te bloque, et tes idées. C'est ce qui transforme "j'ai survolé JEPA" en "j'ai un truc à dire dessus".

---

## PHASE 1 — Fondations (S1–S4, ~150h)

### Semaine 1 — Vision LeCun + maths du DL (~38h)
**Objectif :** comprendre *pourquoi* JEPA existe avant de regarder *comment* il marche.

**Lectures :**
- Yann LeCun, *A Path Towards Autonomous Machine Intelligence* (2022). Manifeste, 60+ pages, lis-le 2 fois.
- Talk YouTube de LeCun à NeurIPS / Collège de France — choisis-en un.
- Blog *"Deep Dive into Yann LeCun's JEPA"* de Rohit Bandaru — meilleure intro vulgarisée.
- LeCun, *A Tutorial on Energy-Based Learning* (2006), sections 1–4.

**Pratique :** mise à niveau PyTorch (60min Blitz, Learn the Basics). Re-coder à la main MLP + une couche d'attention.

**Livrables :** notes structurées (system 1/2, world models, énergie, hiérarchies de prédiction) + notebook PyTorch from scratch.

---

### Semaine 2 — Vision Transformers + masked modeling (~38h)
**Objectif :** maîtriser ViT + MAE, briques de base d'I-JEPA.

**Papiers :**
1. *Attention is All You Need* (Vaswani et al., 2017)
2. *An Image is Worth 16×16 Words* (ViT, Dosovitskiy et al., 2020)
3. *Masked Autoencoders Are Scalable Vision Learners* (MAE, He et al., 2021) — **crucial**, I-JEPA = MAE en latent space
4. *BEiT* (Bao et al., 2021)

**Pratique :** ViT-Tiny en PyTorch (~300 lignes), entraîné sur CIFAR-10. `timm` pour comparer.

**Livrables :** ViT-Tiny qui tourne + note 1 page "Pourquoi MAE marche mieux que reconstruction pixel ?"

---

### Semaine 3 — SSL contrastif (~38h)
**Papiers :**
1. *SimCLR* (Chen et al., 2020)
2. *MoCo v2/v3* (He et al.)
3. *InfoNCE / CPC* (van den Oord et al., 2018)
4. *SSL Cookbook* (Balestriero, Ibrahim et al., 2023) — référence pédagogique ultime

**Pratique :** reproduire SimCLR sur CIFAR-10, évaluer par linear probe.

**Livrables :** notebook + courbe linear probe + note "Pourquoi le contrastif demande des batchs énormes ?"

---

### Semaine 4 — SSL non-contrastif + DINO (~38h)
**Objectif :** comprendre BYOL/DINO/VICReg, vrais ancêtres de JEPA.

**Papiers (par ordre) :**
1. *BYOL* (Grill et al., 2020) — magie du stop-gradient + EMA
2. *SimSiam* (Chen & He, 2021)
3. *VICReg* (Bardes, Ponce, LeCun, 2022) — Bardes = futur auteur de V-JEPA
4. *Barlow Twins* (Zbontar et al., 2021)
5. *DINO* (Caron et al., 2021) puis *DINOv2* (Oquab et al., 2023)

**Concept critique : le "representation collapse"** — pourquoi le modèle veut tricher en sortant un vecteur constant, et comment chacun l'évite. C'est *le* problème central de JEPA.

**Livrables :** tableau comparatif des 5–6 méthodes (objectif, anti-collapse, complexité, perf) + blog post 3 pages "Du contrastif aux JEPAs".

---

## PHASE 2 — Cœur JEPA (S5–S8, ~150h)

### Semaine 5 — I-JEPA : lecture profonde (~38h)
**Lectures :**
- *I-JEPA* (Assran et al., CVPR 2023, arXiv 2301.08243). Lecture *très* lente, équation par équation.
- Repo `facebookresearch/ijepa` : `src/models/vision_transformer.py` + la loss.

**Questions à creuser :**
- Pourquoi context encoder + target encoder + predictor ?
- Pourquoi target encoder = EMA du context encoder ?
- Pourquoi masquage multi-block (et pas aléatoire à la MAE) ?

**Livrables :** notes annotées du papier + schéma d'architecture dessiné à la main.

---

### Semaine 6 — Reproduction I-JEPA (~38h)
**Tâches :**
- Cloner `facebookresearch/ijepa`, lire chaque fichier
- Reproduire l'entraînement sur **CIFAR-100** ou **ImageNet-100** (ImageNet-1k complet hors budget)
- Linear probe, comparer avec MAE et DINOv2 préentraînés
- Bonus : ablation (modifier la stratégie de masquage)

**Compute :** SNU dispose normalement d'un cluster GPU bien fourni — *demande dès la semaine 1 l'accès au cluster du labo et au cluster central SNU*. Sinon Colab Pro suffit en petit modèle.

**Livrables :** repo GitHub + rapport d'expé 5 pages.

---

### Semaine 7 — V-JEPA + Point-JEPA (~38h)
**Cette semaine est doublement importante** : V-JEPA pour l'axe USV/perception vidéo, **Point-JEPA pour l'axe ship CAD** (point clouds, qui est l'expertise historique du Pr. Kim).

**Lectures vidéo :**
1. *V-JEPA* (Bardes et al., 2024, arXiv 2404.08471)
2. *MC-JEPA* (Bardes et al., 2023) — motion + content
3. Comparer mentalement avec *VideoMAE V2*

**Lectures 3D (axe naval) :**
1. *Point-JEPA* (Saito, Kudeshia, Poovvancheri, WACV 2025, arXiv 2404.16432). Repo : `Ayumu-J-S/Point-JEPA`
2. *3D-JEPA* (Hu et al., 2024, arXiv 2409.15803) — multi-block sampling sur point clouds
3. Pour le contexte 3D SSL : *Point-MAE*, *Point2Vec*

**Pratique :** charger un modèle V-JEPA préentraîné, extraire features sur quelques vidéos (idéalement vidéos de mer si tu en trouves). Pour Point-JEPA : tourner le repo sur ShapeNet, voir les représentations apprises sur des objets 3D.

**Livrables :** note "I-JEPA vs V-JEPA vs Point-JEPA : ce qui change selon la modalité" + 2 notebooks démo.

---

### Semaine 8 — V-JEPA 2 + World Models pour la robotique (~38h)
**Objectif :** comprendre le grand saut de juin 2025 — JEPA → world model exploitable pour planifier.

**Lectures :**
1. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* (Assran et al., juin 2025, arXiv 2506.09985). **LE papier flagship.** V-JEPA 2.1 est sortie en mars 2026 sur HuggingFace.
2. *World Models* (Ha & Schmidhuber, 2018)
3. *DreamerV3* (Hafner et al., 2023)
4. **Bonus naval :** repenser à comment ce framework s'applique à un USV. Préentraînement action-free sur de la vidéo de mer + fine-tuning action-conditioned sur des trajectoires d'USV → planning par MPC.

**Concepts à intégrer :**
- Pré-entraînement action-free sur 1M+ heures de vidéo
- Fine-tuning action-conditioned sur ~62h de robot data (Droid)
- Model Predictive Control en latent space → planning zero-shot
- Trois benchmarks publiés : IntPhys 2, MVPBench, CausalVQA

**Pratique :** cloner `facebookresearch/vjepa2`, charger un checkpoint via HuggingFace, tourner `vjepa2_demo.ipynb` (extraction features + classification d'action), lire `energy_landscape_example.ipynb`.

**Livrables :** synthèse "V-JEPA 2 → V-JEPA 2-AC → MPC : la chaîne complète" + notebook qui tourne + **note de proposition de mini-projet** à discuter avec le Pr. Kim au début de la S9.

---

## PHASE 3 — Théorie, frontières, projet personnel (S9–S13, ~180h)

### Semaine 9 — Théorie + variantes + cartographie (~38h)
**Lectures :**
1. *LeJEPA* (Balestriero & LeCun, nov. 2025, arXiv 2511.08544) — **incontournable**, théorie propre, objectif SIGReg sans stop-gradient ni EMA
2. *Var-JEPA* (mars 2026, arXiv 2603.20111) — pont prédictif/génératif
3. *C-JEPA* (Mo & Tong, 2024) — JEPA + VICReg
4. *Rectified LpJEPA* — sparsité
5. **Survey :** Brotee et al., *A Survey on JEPA and World Models* (nov. 2025)

**Bonus naval :** lire 2–3 publications récentes du Pr. Kim et de son labo pour identifier les jeux de données et les benchmarks qu'ils utilisent (très probablement DRoad, leur propre dataset USV, données de l'industrie coréenne).

**Livrables :** note théorique 5–7 pages "Le problème du collapse à travers la famille JEPA" + figure récap des variantes (modalité × technique anti-collapse × année).

---

### Semaine 10 — Multi-modal + cadrage final du mini-projet (~38h)
**Lectures :**
1. *VL-JEPA* (Chen et al., déc. 2025, arXiv 2512.10942) — vision-langage
2. *T-JEPA* (Thimonier et al., ICLR 2025) — tabulaire (utile pour les données de simu CFD ?)
3. *A-JEPA* (Fei et al., 2024) — audio (utile pour la signature acoustique des navires ?)
4. Graph JEPA (Skenderi et al., 2023)

**Cette semaine, fixe ton sujet de mini-projet** avec le Pr. Kim. Quatre options orientées maritime ci-dessous (par ordre de difficulté/risque croissant). Choisis-en *une*.

**Livrables :** carte conceptuelle "JEPA across modalities" + **proposition de mini-projet finalisée** (1–2 pages : question de recherche, méthode, données, métriques, plan expé).

---

### Semaines 11–12 — Mini-projet de recherche (~80h)

#### Option A — Point-JEPA pour les hull forms (CAD naval)
**Le sujet le plus aligné avec l'expertise historique du Pr. Kim.** Prendre un dataset de coques de navires en représentation point cloud (ShipGen, dataset MARIN, ou les hull forms générées par les outils du labo). Entraîner Point-JEPA dessus en self-supervised. Évaluer sur tâches downstream :
- Classification de type de coque
- Régression vers une métrique de performance (drag, displacement)
- Few-shot learning sur des hulls rares (catamarans, brise-glaces)

**Question :** un préentraînement Point-JEPA sur des hulls améliore-t-il la prédiction de performance par rapport à un Point-MAE ou un Point2Vec ?

**Pourquoi c'est bien :** dataset accessible, GPU raisonnable, croise les deux expertises du labo (CAD + DL), publication possible à un workshop ICCAS ou ASME OMAE.

#### Option B — V-JEPA 2 features pour la perception USV
Récupérer un dataset de vidéos marines (Singapore Maritime Dataset, MaSTr1325, ou les données embarquées du labo si elles existent). Utiliser V-JEPA 2 préentraîné comme feature extractor. Entraîner des sondes (probes) légères pour :
- Détection d'horizon / segmentation mer/ciel
- Détection d'autres navires
- Estimation de la rugosité de surface (état de la mer)

Comparer avec DINOv2 et VideoMAE V2.

**Question :** la "compréhension du monde physique" de V-JEPA 2, apprise sur de la vidéo terrestre, transfère-t-elle au domaine marin (sans fine-tuning) ?

**Pourquoi c'est bien :** directement applicable au projet Autonomous Ship Technology du labo. Beaucoup de datasets publics. Pas besoin d'entraîner de gros modèles, juste les sondes.

#### Option C — JEPA temporel pour la prédiction de vagues
Le labo a publié *"A Univariate and Multivariate Machine Learning Approach for Prediction of Significant Wave Height"* (Domala & Kim, Oceans 2022). Refaire le benchmark mais avec un objectif JEPA temporel : prédire les embeddings futurs d'une série temporelle de hauteur de vagues à partir d'un contexte passé.

**Question :** un préentraînement JEPA améliore-t-il la prédiction de Hs par rapport aux LSTM/Transformer entraînés en supervisé direct ?

**Pourquoi c'est bien :** alignement parfait avec les publications récentes du labo. Données publiques (NOAA, KMA buoy data). Calcul léger.

#### Option D — V-JEPA 2-AC en simulateur marin (ambitieux)
Utiliser un simulateur d'USV (Gazebo + UUV Simulator, ou Stonefish) pour collecter des trajectoires (état, action). Adapter V-JEPA 2-AC à ce domaine, faire du planning par MPC en latent space, comparer avec une baseline MPC classique.

**Question :** un world model V-JEPA 2-AC peut-il faire du planning d'évitement d'obstacles aussi bien qu'une approche MPC + dynamique connue ?

**Pourquoi c'est risqué mais ambitieux :** très aligné avec la direction du labo (autonomous ships), mais demande de mettre en place toute la stack simu + RL + JEPA en 2 semaines, ce qui est tendu pour un 1A. À ne tenter que si tu as déjà un encadrant doctorant qui te guide à temps plein dessus.

**Ma reco :** **Option B** comme valeur sûre, **Option A** si tu veux jouer la carte expertise-encadrant, **Option C** si tu préfères les séries temporelles. Évite l'Option D sauf si le Pr. Kim te la propose.

**Livrables fin S12 :** code GitHub propre + résultats quantitatifs + premier draft du rapport.

---

### Semaine 13 — Rédaction et présentation (~40h)
**Livrables finaux :**
1. **Rapport de recherche** 15–25 pages (format article ML : intro, related work, méthode, expés, discussion, limites, conclusion)
2. **Présentation** de soutenance 20–30 min
3. **Repo GitHub** public avec README de reproduction et licence
4. **Blog post** vulgarisé (Medium / Substack) — c'est ce qui rend ton travail visible

---

## Ressources permanentes

### Papiers fondateurs (à avoir en PDF, annotés)
- LeCun 2022 (position paper)
- I-JEPA 2023, V-JEPA 2024, **V-JEPA 2 2025** (flagship)
- **Point-JEPA WACV 2025** (axe CAD du labo)
- LeJEPA nov. 2025 (théorie)
- Survey Brotee nov. 2025

### Code
- `facebookresearch/ijepa`
- `facebookresearch/jepa` (V-JEPA v1)
- `facebookresearch/vjepa2` (+ V-JEPA 2.1)
- `Ayumu-J-S/Point-JEPA`
- HuggingFace : `facebook/vjepa2-*`

### Datasets maritimes utiles
- **MaSTr1325** (segmentation maritime)
- **Singapore Maritime Dataset** (détection navires)
- **MID (Marine Image Dataset)**
- **DRoad** (le dataset ROS du labo, à demander)
- **NOAA/KMA buoy data** (séries de wave height)
- **ShipGen** (paramétrique hull forms)
- **ShapeNet** (pour Point-JEPA)

### Suivi de l'actu
- arXiv-sanity (filtre "JEPA")
- X/Twitter : Yann LeCun, Adrien Bardes, Nicolas Ballas, Randall Balestriero, Mido Assran
- Newsletters : Gonzo ML, TuringPost
- Conférences : NeurIPS, ICML, CVPR, ICLR. Pour le maritime : ICCAS, ASME OMAE, OCEANS, IROS

### Cours pour combler des trous
- Cours NYU "Deep Learning" de LeCun & Canziani (gratuit)
- *Dive into Deep Learning* (d2l.ai)
- SSL Cookbook (Balestriero et al., 2023)

---

## Spécificités du contexte SNU

1. **Demande l'accès GPU dès la S1.** Le cluster du dllab + le cluster central SNU sont normalement bien dotés. Ne te bloque pas en attendant.
2. **Demande à participer aux séminaires hebdo du labo.** À SNU les "research meetings" hebdomadaires sont la norme — assister et présenter même un état d'avancement de 5 min t'apporte énormément.
3. **Identifie un doctorant senior comme mentor de proximité.** Le Pr. Kim est très occupé ; un doctorant à temps plein qui peut répondre à tes questions à 14h fait toute la différence. Repère-le dès les premières réunions.
4. **Langue de travail :** lecture en anglais (papiers), écriture probablement en anglais (publi, rapport), oral souvent en coréen entre membres. Si tu ne parles pas coréen, demande à présenter en anglais — c'est extrêmement courant à SNU et bien accepté.
5. **Industriels coréens :** le labo a probablement des liens avec HD Hyundai (ex-Hyundai Heavy), Samsung Heavy Industries, HMM, K-Shipbuilding. Ton mini-projet peut potentiellement utiliser leurs jeux de données — demande.
6. **Calendrier :** vérifie les jours fériés coréens (Chuseok, Seollal selon le moment de ton stage) pour ajuster les semaines de travail.

---

## Conseils pratiques

1. **Tu seras perdu en S1–S2. C'est normal.** Lire un papier ML après 1A de prépa, c'est dur. Force-toi à finir, tu reviendras dessus en S5–S6 et là ce sera limpide.
2. **Code dès la S1.** Lire sans coder = oublier en 3 jours.
3. **Ne saute pas les fondations.** Beaucoup veulent attaquer V-JEPA 2 directement. Sans BYOL/DINO/MAE en tête, tu ne fais que recopier.
4. **Limite le scope du mini-projet.** En 2 semaines tu n'inventes pas une nouvelle architecture. Une étude empirique propre et bien analysée vaut mille fois mieux qu'une "innovation" mal exécutée.
5. **Écris au fil de l'eau.** Le rapport final s'agrège à partir de tes notes hebdo, pas écrit en panique en S13.
6. **Réunion bilatérale toutes les 2 semaines minimum** avec le Pr. Kim ou son doctorant senior. Prépare 3 slides : ce que j'ai fait, ce qui me bloque, ce que je vais faire.
7. **Garde une slide qui s'enrichit chaque semaine** : la "vue d'ensemble" de ton projet. C'est ta soutenance qui se construit toute seule.

Bon courage — tu tombes sur un labo où JEPA et le maritime ont vraiment du sens à se croiser, et ce n'est pas encore un terrain saturé.
