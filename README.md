# Football AI Predictor — version prête au déploiement

Application web Streamlit en français. Elle télécharge automatiquement les résultats de football-data.co.uk (saisons 2024/25, 2025/26 et 2026/27 lorsqu'elles sont disponibles), entraîne un modèle Dixon–Coles et permet d'activer trois agents IA et un superviseur via OpenRouter.

## Ce que fait l'application
- Sélection d'un championnat (Premier League, Liga, Bundesliga, Serie A, Ligue 1, Championship).
- Téléchargement automatique des résultats récents ; pas besoin d'importer manuellement les CSV.
- Sélection des matchs futurs trouvés dans les fichiers, ou choix manuel de deux équipes.
- Probabilités 1X2, over/under 2,5, BTTS, buts attendus et scores probables via Dixon–Coles.
- Trois rôles IA (statistique, tactique, actualités) puis un superviseur qui confronte les conclusions.
- Recherche web Tavily facultative et export du rapport.

## Mise en service la plus simple : Streamlit Community Cloud
Cette application web ne s'installe pas comme un APK Android. Elle doit être hébergée une première fois ; ensuite, tu l'ouvres dans Chrome sur ton téléphone et tu peux l'ajouter à l'écran d'accueil.

1. Crée un compte GitHub et un compte sur https://share.streamlit.io/.
2. Décompresse cette archive et place les fichiers `app.py` et `requirements.txt` dans un nouveau dépôt GitHub.
3. Sur Streamlit Community Cloud, choisis **Deploy an app**, sélectionne le dépôt, puis `app.py`.
4. L'application se lance avec une URL publique. Ouvre cette URL sur Android avec Chrome > menu ⋮ > **Ajouter à l'écran d'accueil**.
5. Pour les analyses multi-IA, crée une clé sur https://openrouter.ai/ et saisis-la dans la barre latérale de l'application. Les appels de modèles peuvent être payants. Sans cette clé, le moteur Dixon–Coles fonctionne quand même.
6. Facultatif : ajoute une clé Tavily (https://tavily.com/) pour la recherche d'actualités. Sans Tavily, l'agent ne doit pas prétendre avoir fait une recherche web.

## Déploiement Docker (facultatif)
```bash
docker build -t football-ai-predictor .
docker run -p 8501:8501 football-ai-predictor
```
Puis ouvrir `http://localhost:8501`.

## Limites à connaître
- Les données de résultats et calendriers viennent de football-data.co.uk ; disponibilité variable selon championnat et saison.
- Le fichier ne fournit pas toujours les rencontres futures. Dans ce cas, sélectionner les équipes manuellement.
- La Ligue des Champions n'est pas proposée comme championnat d'entraînement : les données de ligue nationale servent à estimer les forces des équipes, mais ce n'est pas un modèle spécifique à l'UCL.
- Les agents IA nécessitent une clé OpenRouter et peuvent engendrer des frais. Tavily est facultatif et peut également être payant.
- Une analyse multi-agents n'est pas une garantie d'exactitude. Valider les performances avec des backtests avant toute utilisation financière.
