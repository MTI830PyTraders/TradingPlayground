<!--
%  Intro
%    * Déroulement de la présentation
%    * Expliquer le contexte (prédiction des stocks avec l'analyse de sentiments)
%  Prolog de l'histoire : Comment on s'est fait abordé
%    * Rencontre avec Françis
%    * Explication de l'hypothèse de base (Stock price * Sentiments = DCF)
%  Histoire : Conclusion de ce qu'on va faire.  Hypothèse de base. Discounted cash flow.
%  1ère itération : sentiment140
%    * Décrire sentiment140
%    * Expliquer ce qu'on a fait avec sentiment140, montrer les résultats
%  2e  fuck
%    * Expliquer que ça prend du temps et que c'est pas fiable
%  3e  carl à la rescousse : finsents + sharadar + Free Cash + Quandl
%    * Expliquer c'est quoi finsents, sharadar + FCF
%    * Préparation des données. Pandas + Xarray. Nettoyage avec Bash
%  4e  LSTM
%    * Explication du LSTM
%    * Expliquer les paramètres
%  5e  Random Forest
%    * Expliquer Random Forest
%  6e  Résultats
%    * Montrer les résultats avec des graphiques
%  7e  Conclusion
%  8e Travaux futurs
-->

# section Préambule
## sous-section Déroulement de la présentation
### frame Déroulement de la présentation
* Introduction
* Présentation de nos experts
* Premières expériences
* Méthodes
* Résultats
* Conclusion

## sous-section Introduction
### frame Introduction
### graph xkcd

### frame Objectif
* Faire la prédiction de la valeur d'une action dans le futur avec le deep learning
* Utiliser l'analyse de sentiments
* Utiliser les différentes informations disponibles sur une compagnie

## sous-section Présentation de nos experts
### frame Présentation de Françis
* Partenaire d'affaire de Mike
* Ancien étudiant et diplômé du MILA
* Expert en Machine Learning et en investissement à la bourse
* a proposé l'hypothèse suivante:
* Stock price * Sentiments = Discounted Cash Flow
### frame Présentation de Carl 
* Détient un MBA du HEC
* Fondateur de la startup Evovest
* Expert en finance et en machine learning
* 
## sous-section Premières expériences
### frame Expérimentation avec sentiment140
* Dataset de 1.6 millions de tweets disponible publiquement sur internet
* Publié par des étudiants en science informatique de l'université de Stanford
* Premier contact avec le deep learning
* Nous avons été capables d'attribuer un sentiment à une phrase grâce au modèle
### frame Expérimentation avec sentiment140 (2)
* Résultats peu fiables
* Nécessite beaucoup plus de travail pour valider le modèle
* Nécessite encore plus de travail pour mettre la main sur un autre dataset de Twitter
## sous-section Recommendations de Carl
### frame FinSents 
* Génère une valeur de sentiment par jour par compagnie
* La valeur provient de différentes sources:
  * Twitter
  * Les journaux
  * Différents réseaux sociaux
  * Les blogues
* Offre d'autres colonnes tels le News Buzz et le News Volume
### frame FinSents 2
* Couvre les sentiments de dizaines de milliers de compagnies à travers le monde:
  * 15 000 nord-américaines
  * 8 000 européennes
  * 4 000 japonaises
  * 14 000 asiatiques à l'exclusion du Japon
  * 3 000 Australie / Nouvelle-Zélande
  * 1 000 sud-américaines
### frame Sharadar
* Contient de l'information financière sur plus de 12 000 compagnies américaines
* Chaque compagnie possède plus d'une centaine de colonnes d'information financière
* Peut fournir de l'information sur une compagnie jusqu'à 20 ans dans le passé
### frame Datasets
* Combinaison possible des datasets avec la date et le ticker
* Datasets disponibles sur Quandl
* Possibilité d'utiliser l'API de Quandl
* Possibilité de télécharger les datasets complets au format CSV
* Datasets payants au coût de 50$ par mois

# section Méthodes
## sous-section Préparation des données
### frame Datasets
### frame Fusion des données 
## sous-section Entrainement d'un modèle avec LSTM 
### frame Présentation du LSTM 
* 
* 
* 
# section Résultats
## sous-section Résultats
# section Conclusion
## sous-section Conclusion
### frame Conclusion
### graph xkcd
## sous-section Travaux futur
### frame Travaux futur
* Évaluer la pertinence des autres attributs du dataset
* Tester avec plus de tickers
* Utiliser plusieurs tickers pour entrainer un même modèle
* Modifier les paramètres d'entrainement du modèle
* Entrainer d'autres types de modèles et comparer les résultats
