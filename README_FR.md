# AutoGPT: Construisez, Déployez et Exécutez des Agents IA

[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoGPT** est une plateforme puissante qui vous permet de créer, déployer et gérer des agents IA continus automatisant des flux de travail complexes.

## Options d'hébergement 
   - Téléchargez pour l'auto-héberger
   - [Rejoignez la liste d'attente](https://bit.ly/3ZDijAI) pour la version bêta hébergée sur le cloud 

## Comment Configurer pour l'Auto-Hébergement
> [!NOTE]
> La configuration et l'hébergement de la plateforme AutoGPT sont des processus techniques. 
> Si vous préférez quelque chose de plus simple, nous vous recommandons de [rejoindre la liste d'attente](https://bit.ly/3ZDijAI) pour la version bêta hébergée sur le cloud.

https://github.com/user-attachments/assets/d04273a5-b36a-4a37-818e-f631ce72d603

Ce tutoriel suppose que vous avez Docker, VSCode, git et npm installés.

### 🧱 Frontend AutoGPT

Le frontend AutoGPT est l'endroit où les utilisateurs interagissent avec notre puissante plateforme d'automatisation IA. Il offre plusieurs façons d'engager et de tirer parti de nos agents IA. C'est l'interface où vous concrétisez vos idées d'automatisation IA :

   **Agent Builder :** Pour ceux qui veulent personnaliser, notre interface intuitive sans code vous permet de concevoir et configurer vos propres agents IA.
   
   **Gestion des flux de travail :** Créez, modifiez et optimisez facilement vos flux de travail d'automatisation. Vous construisez votre agent en connectant des blocs, chaque bloc exécutant une seule action.
   
   **Contrôles de Déploiement :** Gérez le cycle de vie de vos agents, des tests à la production.
   
   **Agents prêts à l'emploi :** Pas envie de créer ? Sélectionnez simplement dans notre bibliothèque d'agents préconfigurés et mettez-les immédiatement au travail.
   
   **Interaction avec les agents :** Que vous ayez créé le vôtre ou que vous utilisiez des agents préconfigurés, exécutez-les et interagissez avec eux facilement via notre interface utilisateur conviviale.

   **Surveillance et Analyse :** Suivez la performance de vos agents et obtenez des informations pour améliorer continuellement vos processus d'automatisation.

[Consultez ce guide](https://docs.agpt.co/server/new_blocks/) pour apprendre à construire vos propres blocs personnalisés.

### 💽 Serveur AutoGPT

Le serveur AutoGPT est le moteur de notre plateforme. C'est ici que vos agents s'exécutent. Une fois déployés, les agents peuvent être déclenchés par des sources externes et fonctionner en continu. Il contient tous les composants essentiels pour assurer le bon fonctionnement d'AutoGPT.

   **Code Source :** La logique centrale qui fait tourner nos agents et les processus d'automatisation.
   
   **Infrastructure :** Des systèmes robustes assurant des performances fiables et évolutives.
   
   **Marketplace :** Un marché complet où vous pouvez trouver et déployer une large gamme d'agents préconstruits.

### 🐙 Exemples d'agents

Voici deux exemples de ce que vous pouvez faire avec AutoGPT :

1. **Générer des vidéos virales à partir de sujets tendance**
   - Cet agent lit les sujets sur Reddit.
   - Il identifie les sujets en tendance.
   - Il crée ensuite automatiquement une vidéo courte basée sur le contenu.

2. **Identifier les meilleures citations à partir de vidéos pour les réseaux sociaux**
   - Cet agent s'abonne à votre chaîne YouTube.
   - Lorsque vous postez une nouvelle vidéo, il la transcrit.
   - Il utilise l'IA pour identifier les citations les plus percutantes afin de générer un résumé.
   - Ensuite, il écrit un post à publier automatiquement sur vos réseaux sociaux.

Ces exemples ne sont qu'un aperçu de ce que vous pouvez réaliser avec AutoGPT ! Vous pouvez créer des flux de travail personnalisés pour construire des agents adaptés à tous vos cas d'utilisation.

---
### Mission et Licence
Notre mission est de fournir les outils nécessaires pour que vous puissiez vous concentrer sur ce qui compte :

- 🏗️ **Construire** - Posez les bases de quelque chose d'incroyable.
- 🧪 **Tester** - Affinez votre agent à la perfection.
- 🤝 **Déléguer** - Laissez l'IA travailler pour vous et donnez vie à vos idées.

Soyez partie de la révolution ! **AutoGPT** est là pour rester, à l'avant-garde de l'innovation en IA.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contribuer](CONTRIBUTING.md)**

**Licence :**

Licence MIT : La majorité du dépôt AutoGPT est sous la licence MIT.

Licence Polyform Shield : Cette licence s'applique au dossier autogpt_platform.

Pour plus d'informations, consultez https://agpt.co/blog/introducing-the-autogpt-platform

---
## 🤖 AutoGPT Classique
> Ci-dessous, des informations sur la version classique d'AutoGPT.

**🛠️ [Construire votre propre agent - Démarrage rapide](classic/FORGE-QUICKSTART.md)**

### 🏗️ Forge

**Forgez votre propre agent !** &ndash; Forge est une boîte à outils prête à l'emploi pour construire votre propre application d'agent. Elle gère la majorité du code standard, vous permettant de canaliser toute votre créativité dans les aspects qui distinguent *votre* agent. Tous les tutoriels sont disponibles [ici](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec). Les composants de [`forge`](/classic/forge/) peuvent également être utilisés individuellement pour accélérer le développement et réduire le code standard dans votre projet d'agent.

🚀 [**Commencer avec Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md) &ndash;
Ce guide vous guidera à travers le processus de création de votre propre agent et l'utilisation de l'interface utilisateur et des benchmarks.

📘 [En savoir plus](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge) sur Forge

### 🎯 Benchmark

**Mesurez la performance de votre agent !** Le `agbenchmark` peut être utilisé avec tout agent qui prend en charge le protocole d'agent, et l'intégration avec le [CLI] du projet le rend encore plus facile à utiliser avec AutoGPT et les agents basés sur Forge. Le benchmark offre un environnement de test rigoureux. Notre cadre permet des évaluations de performance autonomes et objectives, garantissant que vos agents sont prêts pour une action réelle.

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) sur Pypi
&ensp;|&ensp;
📘 [En savoir plus](https://github.com/Significant-Gravitas/AutoGPT/blob/master/benchmark) sur le Benchmark

### 💻 Interface Utilisateur

**Facilitez l'utilisation des agents !** Le `frontend` vous offre une interface conviviale pour contrôler et surveiller vos agents. Il se connecte aux agents via le [protocole d'agent](#-agent-protocol), garantissant la compatibilité avec de nombreux agents de notre écosystème et d'ailleurs.

Le frontend fonctionne prêt à l'emploi avec tous les agents du dépôt. Utilisez simplement le [CLI] pour exécuter l'agent de votre choix !

📘 [En savoir plus](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend) sur l'interface utilisateur

### ⌨️ CLI

[CLI]: #-cli

Pour faciliter l'utilisation de tous les outils offerts par le dépôt, un CLI est inclus à la racine du dépôt :

```shell
$ ./run
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Affiche cette aide et quitte.

Commands:
  agent      Commandes pour créer, démarrer et arrêter des agents
  benchmark  Commandes pour démarrer le benchmark et lister les tests et catégories
  setup      Installe les dépendances nécessaires pour votre système.
```


Clonez simplement le dépôt, installez les dépendances avec `./run setup`, et tout devrait bien se passer !

## 🤔 Questions ? Problèmes ? Suggestions ?

### Obtenez de l'aide - [Discord 💬](https://discord.gg/autogpt)

[![Rejoignez-nous sur Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

Pour signaler un bug ou demander une fonctionnalité, créez un [Issue GitHub](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose). Assurez-vous qu'il n'y a pas déjà un sujet ouvert pour le même problème.

## 🤝 Projets associés

### 🔄 Protocole Agent

Pour maintenir une norme uniforme et assurer une compatibilité transparente avec de nombreuses applications actuelles et futures, AutoGPT utilise le [protocole agent](https://agentprotocol.ai/) développé par la AI Engineer Foundation. Ce protocole standardise les voies de communication entre votre agent et l'interface utilisateur ainsi que le banc d'essai.

---

<p align="center">
<a href="https://star-history.com/#Significant-Gravitas/AutoGPT">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
    <img alt="Graphique de l'historique des étoiles" src="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
  </picture>
</a>
</p>
