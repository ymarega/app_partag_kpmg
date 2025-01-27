import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import ruptures as rpt

import plotly.graph_objects as go
from plotly.subplots import make_subplots


## Entrainement 5 modèles de machine learning pour la détection de fuites en minimisant l'accuracy
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Configuration de la page
#st.set_page_config(layout="wide")


st.title('Détection fuite dans le réseau de distribution d\'eau potable - Saint-Esprit')



st.header('1. Analyse exploratoire des données')

st.subheader('1.1 Débit de distribution d\'eau potable à Saint-Esprit')


#debit_stesprit_heure_pivot = pd.read_csv("C:/Users/ymarega/Dropbox/Nordicite (BD)/saint_esprit/debit_stesprit_heure_pivot.csv", index_col=0, parse_dates=True)


debit_stesprit_heure_pivot = pd.read_csv('https://www.dropbox.com/scl/fi/h0v5gj84d947fq6n4ptru/debit_stesprit_heure_pivot.csv?rlkey=nhd6plh9fn9xf7e2cx37if98v&st=v9phhewm&dl=1', index_col=0, parse_dates=True)
st.write(debit_stesprit_heure_pivot.head(10))
fig = go.Figure()
fig.add_trace(go.Scatter(x=debit_stesprit_heure_pivot.index, y=debit_stesprit_heure_pivot['00h-01h'], mode='lines', name='00h-01h'))
fig.add_trace(go.Scatter(x=debit_stesprit_heure_pivot.index, y=debit_stesprit_heure_pivot['01h-02h'], mode='lines', name='01h-02h'))
fig.add_trace(go.Scatter(x=debit_stesprit_heure_pivot.index, y=debit_stesprit_heure_pivot['02h-03h'], mode='lines', name='02h-03h'))
fig.add_trace(go.Scatter(x=debit_stesprit_heure_pivot.index, y=debit_stesprit_heure_pivot['03h-04h'], mode='lines', name='03h-04h'))
fig.update_layout(title='Moyenne débit Saint-Esprit par heure', xaxis_title='Jour', yaxis_title='Débit st esprit')
# afficher le graphique dans l'interface
st.plotly_chart(fig)

## Afficher les statistiques descriptives des données entre 00h-01h, 01h-02h, 02h-03h, 03h-04h
st.subheader('Statistiques descriptives des données')
st.write(debit_stesprit_heure_pivot[['00h-01h','01h-02h','02h-03h','03h-04h']].describe())

## Mettre les donnéées qui ne sont pas entre 4 et 40 a NaN pour toutes les heures
debit_stesprit_heure_pivot[(debit_stesprit_heure_pivot < 4) | (debit_stesprit_heure_pivot > 40)] = None
fig = go.Figure()
fig.add_trace(go.Scatter(x=debit_stesprit_heure_pivot.index, y=debit_stesprit_heure_pivot['00h-01h'], mode='lines', name='00h-01h'))
fig.add_trace(go.Scatter(x=debit_stesprit_heure_pivot.index, y=debit_stesprit_heure_pivot['01h-02h'], mode='lines', name='01h-02h'))
fig.add_trace(go.Scatter(x=debit_stesprit_heure_pivot.index, y=debit_stesprit_heure_pivot['02h-03h'], mode='lines', name='02h-03h'))
fig.add_trace(go.Scatter(x=debit_stesprit_heure_pivot.index, y=debit_stesprit_heure_pivot['03h-04h'], mode='lines', name='03h-04h'))
fig.update_layout(title='Moyenne débit Saint-Esprit par heure', xaxis_title='Jour', yaxis_title='Débit st esprit')
# afficher le graphique dans l'interface
st.plotly_chart(fig)


## Afficher les statistiques descriptives des données entre 00h-01h, 01h-02h, 02h-03h, 03h-04h
st.subheader('Statistiques descriptives des données')
st.write(debit_stesprit_heure_pivot[['00h-01h','01h-02h','02h-03h','03h-04h']].describe())





st.subheader('Choix des heures')
heure = st.selectbox('Choisir l\'heure à visualiser', ['00h-01h', '01h-02h', '02h-03h', '03h-04h', '04h-05h', 
                                                               '05h-06h', '06h-07h', '07h-08h', '08h-09h', 
                                                               '09h-10h', '10h-11h', '11h-12h', '12h-13h', 
                                                               '13h-14h', '14h-15h', '15h-16h', '16h-17h', 
                                                               '17h-18h', '18h-19h', '19h-20h', '20h-21h', 
                                                               '21h-22h', '22h-23h', '23h-00h'])




data = debit_stesprit_heure_pivot.copy()
data['year']= data.index.year

# Création du graphique avec Plotly
fig = go.Figure()

for year in data['year'].unique():
    yearly_data = data[data['year'] == year]
    fig.add_trace(go.Scatter(
        x=yearly_data.index.dayofyear,
        y=yearly_data[heure],
        mode='lines',
        name=str(year) + ' ' + heure
    ))

fig.update_layout(
    title='Variation des débits st esprit par année',
    xaxis_title='Jour de l\'année',
    yaxis_title='Débit st esprit',
    legend_title='Année'
)


# Affichage du graphique avec Streamlit
st.plotly_chart(fig)

# Calcul des statistiques descriptives pour chaque année à l'heure choisie
stats = data.groupby('year')[heure].describe()

# Affichage des statistiques descriptives
st.subheader(f'Statistiques descriptives pour l\'heure {heure}')
st.write(stats)


st.subheader('Choix de la fenêtre de lissage')
window_size = st.slider('Choisir la fenêtre de lissage - Moving Average', min_value=1, max_value=30, value=7)


# Création du graphique avec Plotly
fig = go.Figure()

for year in data['year'].unique():
    yearly_data = data[data['year'] == year]
    fig.add_trace(go.Scatter(
        x=yearly_data.index.dayofyear,
        ## Calcul de la moyenne mobile centrée
        y=yearly_data[heure].rolling(window=window_size, center=True).mean().bfill(),
        mode='lines',
        name=str(year) + ' ' + heure
    ))

fig.update_layout(
    title='Variation des débits st esprit par année',
    xaxis_title='Jour de l\'année',
    yaxis_title='Débit st esprit',
    legend_title='Année'
)


# Affichage du graphique avec Streamlit
st.plotly_chart(fig)


## Grahique de la moyenne debit_stesprit par mois pour une heure donnée
data['month'] = data.index.month
data['year'] = data.index.year
## Pour chaque année et chaque mois, calculer la moyenne du débit st esprit
data_par_mois = data.groupby(['year', 'month'])[heure].mean()


fig = go.Figure()
for year in data['year'].unique():
    yearly_data = data_par_mois.loc[year]
    fig.add_trace(go.Scatter(
        x=yearly_data.index,
        y=yearly_data,
        mode='lines',
        name=str(year)+ ' ' + heure
    ))
st.plotly_chart(fig)








st.header('2. Indicateurs statistiques')


st.subheader("1. Analyse des seuils statiques")

st.write("""
**Principe :**
Cette méthode identifie les anomalies en comparant les valeurs des débits horaires à un seuil fixe. Toute valeur qui dépasse ce seuil prédéfini est considérée comme une anomalie.
""")

st.write("""
**Paramètres utilisés :**
- **Seuil statique (threshold_static)** : 60ème percentile des valeurs de débit.
""")

st.write("""
**Explication :**
Le seuil est défini comme le 60ème percentile des valeurs de débit (`threshold_static = data[heure].quantile(0.6)`). Les valeurs de débit qui dépassent ce seuil sont marquées comme des anomalies, indiquant des périodes où le débit est anormalement élevé par rapport à la majorité des valeurs observées.
""")

st.subheader("2. Analyse des tendances et des moyennes mobiles")

st.write("""
**Principe :**
Cette méthode utilise une moyenne mobile pour lisser les données et détecter des tendances. Une valeur est considérée comme une anomalie si elle est significativement supérieure à la tendance récente.
""")

st.write("""
**Paramètres utilisés :**
- **Fenêtre de la moyenne mobile** : 3 périodes.
- **Seuil d'augmentation** : 1% par rapport à la moyenne mobile précédente.
""")

st.write("""
**Explication :**
Une moyenne mobile est calculée sur une période de 3 heures (`data['moving_avg'] = data[heure].rolling(window=3).mean()`). Si la moyenne mobile actuelle est supérieure de plus de 1% à la moyenne mobile précédente (`moving_avg_leaks = data[data['moving_avg'] > data['moving_avg'].shift(1) * 1.01]`), cela indique une augmentation significative du débit, suggérant une anomalie. Cette méthode permet de détecter des variations graduelles dans les données qui peuvent indiquer des problèmes potentiels.
""")

st.subheader("3. Détection de changements soudains")

st.write("""
**Principe :**
Cette méthode utilise des algorithmes spécialisés (comme PELT) pour détecter des points de rupture où la statistique des débits change soudainement.
""")

st.write("""
**Paramètres utilisés :**
- **Modèle de l'algorithme** : RBF (Radial Basis Function).
- **Pénalité (pen)** : 0.00000001.
""")

st.write("""
**Explication :**
L'algorithme PELT est ajusté sur les valeurs de débit (`algo = rpt.Pelt(model="rbf").fit(signal)`). La prédiction des points de changement utilise une pénalité très faible (`result = algo.predict(pen=0.00000001)`) pour détecter de nombreux changements. Les points où la distribution statistique des valeurs de débit change brusquement sont identifiés, suggérant des anomalies.
""")

st.subheader("4. Analyse des différences journalières")

st.write("""
**Principe :**
Cette méthode identifie les anomalies en analysant les différences journalières des valeurs de débit. Une différence est considérée comme une anomalie si elle dépasse une certaine valeur seuil basée sur la moyenne et l'écart type des différences.
""")

st.write("""
**Paramètres utilisés :**
- **Multiplicateur de l'écart type** : 0.14.
""")

st.write("""
**Explication :**
Les différences journalières sont calculées pour chaque paire de jours consécutifs (`data['daily_diff'] = data[heure].diff()`). Le seuil est déterminé comme la moyenne des différences journalières plus 0.14 fois l'écart type (`threshold_diff = mean_diff + 0.14 * std_diff`). Si la différence journalière dépasse ce seuil, cela indique une anomalie (`diff_leaks = data[data['daily_diff'] > threshold_diff]`). Cette méthode permet de détecter des variations exceptionnelles dans les débits d'un jour à l'autre, suggérant des problèmes ou des changements significatifs dans le système.
""")




# Prepare the dataset
data = debit_stesprit_heure_pivot[[heure]]

## lissage des données par moyenne mobile sur 7 periodes
data[heure] = data[heure].rolling(window=7).mean()



data['date_creation'] = data.index
data = data.sort_values('date_creation')
data['daily_diff'] = data[heure].diff()

# 1. Analyse des seuils statiques
threshold_static = data[heure].quantile(0.92)
static_leaks = data[data[heure] > threshold_static]

# 2. Analyse des tendances et des moyennes mobiles
data['moving_avg'] = data[heure].rolling(window=3).mean()
moving_avg_leaks = data[data['moving_avg'] > data['moving_avg'].shift(1) * 1.03]

# 3. Détection de changements soudains

signal = data[heure].dropna().values
algo = rpt.Pelt(model="rbf").fit(signal)
result = algo.predict(pen=0.3)
changepoints = data.iloc[result[:-1]]

# 4. Analyse des différences journalières
mean_diff = data['daily_diff'].mean()
std_diff = data['daily_diff'].std()
threshold_diff = mean_diff + 0.85* std_diff
diff_leaks = data[data['daily_diff'] > threshold_diff]

# Create a Plotly figure
fig = go.Figure()

# Add traces to the figure
fig.add_trace(go.Scatter(x=data['date_creation'], y=data[heure], mode='lines', name='Débit Saint-Esprit'))
fig.add_trace(go.Scatter(x=static_leaks['date_creation'], y=static_leaks[heure], mode='markers', name='Seuil statique', marker=dict(color='red', symbol='circle')))
fig.add_trace(go.Scatter(x=moving_avg_leaks['date_creation'], y=moving_avg_leaks[heure], mode='markers', name='Moyenne mobile', marker=dict(color='blue', symbol='x')))
fig.add_trace(go.Scatter(x=changepoints['date_creation'], y=changepoints[heure], mode='markers', name='Changement soudain', marker=dict(color='green', symbol='triangle-down')))
fig.add_trace(go.Scatter(x=diff_leaks['date_creation'], y=diff_leaks[heure], mode='markers', name='Différence journalière', marker=dict(color='orange', symbol='square')))
#fig.add_trace(go.Scatter(x=ml_leaks['date_creation'], y=ml_leaks['vol_stesprit_nuit'], mode='markers', name='Machine Learning', marker=dict(color='purple', symbol='diamond')))


# Set the layout of the figure
fig.update_layout(title='', xaxis_title='Date', yaxis_title='Débit Saint-Esprit', legend_title='Méthodes')

# Display the figure using Streamlit
st.plotly_chart(fig)



# Créez une figure avec des subplots 2x2
fig = make_subplots(rows=2, cols=2, subplot_titles=("Débit Saint-Esprit et Seuil statique", "Débit Saint-Esprit et Moyenne mobile", "Débit Saint-Esprit et Changement soudain", "Débit Saint-Esprit et Différence journalière"))

# Premier subplot
fig.add_trace(go.Scatter(x=data['date_creation'], y=data[heure], mode='lines', name='Débit Saint-Esprit entre ' + heure, line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=static_leaks['date_creation'], y=static_leaks[heure], mode='markers', name='Seuil statique', marker=dict(color='red', symbol='circle', size = 3)), row=1, col=1)

# Deuxième subplot
fig.add_trace(go.Scatter(x=data['date_creation'], y=data[heure], mode='lines', name='Débit Saint-Esprit entre ' + heure, line=dict(color='blue')), row=1, col=2)
fig.add_trace(go.Scatter(x=moving_avg_leaks['date_creation'], y=moving_avg_leaks[heure], mode='markers', name='Moyenne mobile', marker=dict(color='red', symbol='x', size = 7)), row=1, col=2)

# Troisième subplot
fig.add_trace(go.Scatter(x=data['date_creation'], y=data[heure], mode='lines', name='Débit Saint-Esprit entre ' + heure, line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=changepoints['date_creation'], y=changepoints[heure], mode='markers', name='Changement soudain', marker=dict(color='green', symbol='triangle-down', size =7)), row=2, col=1)

# Quatrième subplot
fig.add_trace(go.Scatter(x=data['date_creation'], y=data[heure], mode='lines', name='Débit Saint-Esprit entre ' + heure, line=dict(color='blue')), row=2, col=2)
fig.add_trace(go.Scatter(x=diff_leaks['date_creation'], y=diff_leaks[heure], mode='markers', name='Différence journalière', marker=dict(color='orange', symbol='square', size =7)), row=2, col=2)

# Mettre à jour la disposition de la figure
fig.update_layout(title='Détection des fuites par différentes méthodes', xaxis_title='Date', yaxis_title='Débit Saint-Esprit')

# Affichez la figure en utilisant Streamlit
st.plotly_chart(fig)

heures = '02h-04h'
## tracer la figure pour le seuil statique
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['date_creation'], y=data[heure], mode='lines', name='débit entre ' + heures, line=dict(color='blue')))
fig.add_trace(go.Scatter(x=static_leaks['date_creation'], y=static_leaks[heure], mode='markers', name='seuil statique', marker=dict(color='red', symbol='circle')))
fig.update_layout(title='', xaxis_title='Date', yaxis_title='Débit (m3/h)')
st.plotly_chart(fig)

## tracer la figure pour la moyenne mobile
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['date_creation'], y=data[heure], mode='lines', name='débit entre ' + heures, line=dict(color='blue')))
fig.add_trace(go.Scatter(x=moving_avg_leaks['date_creation'], y=moving_avg_leaks[heure], mode='markers', name='moyenne mobile', marker=dict(color='red', symbol='x'))
)
fig.update_layout(title='', xaxis_title='Date', yaxis_title='Débit (m3/h)')
st.plotly_chart(fig)

## tracer la figure pour les changements soudains
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['date_creation'], y=data[heure], mode='lines', name='débit entre ' + heures, line=dict(color='blue')))
fig.add_trace(go.Scatter(x=changepoints['date_creation'], y=changepoints[heure], mode='markers', name='changement soudain', marker=dict(color='green', symbol='triangle-down')))
fig.update_layout(title='', xaxis_title='Date', yaxis_title='Débit (m3/h)')
st.plotly_chart(fig)

## tracer la figure pour les différences journalières
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['date_creation'], y=data[heure], mode='lines', name='débit entre ' + heures, line=dict(color='blue')))
fig.add_trace(go.Scatter(x=diff_leaks['date_creation'], y=diff_leaks[heure], mode='markers', name='différence journalière', marker=dict(color='orange', symbol='square')))
fig.update_layout(title='', xaxis_title='Date', yaxis_title='Débit (m3/h)')
st.plotly_chart(fig)




### Pourcentage de fuites avec les différentes méthodes
static_leaks_count = static_leaks.shape[0]
moving_avg_leaks_count = moving_avg_leaks.shape[0]
changepoints_count = changepoints.shape[0]
diff_leaks_count = diff_leaks.shape[0]
#ml_leaks_count = ml_leaks.shape[0]
print(f'Pourcentage de fuites avec seuil statique: {static_leaks_count / data.shape[0] * 100:.2f}%')
print(f'Pourcentage de fuites avec moyenne mobile: {moving_avg_leaks_count / data.shape[0] * 100:.2f}%')
print(f'Pourcentage de fuites avec changements soudains: {changepoints_count / data.shape[0] * 100:.2f}%')
print(f'Pourcentage de fuites avec différences journalières: {diff_leaks_count / data.shape[0] * 100:.2f}%')
#print(f'Pourcentage de fuites avec machine learning: {ml_leaks_count / data.shape[0] * 100:.2f}%')



## Afficher les pourcentages de fuites avec les différentes méthodes sous forme de tableau
st.subheader('Pourcentage de fuites avec les différentes méthodes')
st.write(pd.DataFrame({
    'Méthode': ['Seuil statique', 'Moyenne mobile', 'Changements soudains', 'Différences journalières'],
    'Pourcentage Anomalie': [static_leaks_count / data.shape[0] * 100, moving_avg_leaks_count / data.shape[0] * 100, changepoints_count / data.shape[0] * 100, diff_leaks_count / data.shape[0] * 100]
}))


st.header('3. Détection de fuites en utilisant le débit de nuit pour étiqutter les données et entrainer un modèle de machine learning')

## 

def developpement_modele(Methodes):
    data['fuite'] = 0
    data.loc[Methodes.index, 'fuite'] = 1
    y = data['fuite']
    #X = pd.read_csv('X.csv', index_col=0, parse_dates=True)
    X = pd.read_csv('https://www.dropbox.com/scl/fi/4qp2sxya9zrp8d1knlxbb/X.csv?rlkey=pjsx9mrtmw0uoyjrugkq1nnfh&st=dhwf72hz&dl=1', index_col=0, parse_dates=True)
    
    df = pd.concat([X, y], axis=1).dropna()
    #df = df.head(700)
    st.dataframe(df)
    X = df.drop('fuite', axis=1)
    y = df['fuite']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'Neural Network': MLPClassifier()
    
    }
    ## Stocker les résultats de chaque modèle
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    ## Afficher les résultats sous forme de tableau
    st.subheader('Résultats des modèles de machine learning')
    st.write(pd.DataFrame(results.items(), columns=['Modèle', 'Accuracy']))


    ## Trouver le modèle avec la meilleure accuracy
    best_model = max(results, key=results.get)
    st.write(f'Le modèle avec la meilleure accuracy est {best_model} avec une accuracy de {results[best_model]:.2f}')

    ## Afficher les résultats de la prédiction du meilleur modèle
    model = models[best_model]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    ## Afficher la matrice de confusion
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    ### Calculer les autres métriques : precision, recall, f1-score¸¸
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    import seaborn as sns
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(2, 1))  # Ajustez la taille de la figure ici
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, annot_kws={"size": 4})

    # Ajuster les étiquettes pour les rendre plus compactes
    ax.set_xlabel('Prédictions', fontsize=3)
    ax.set_ylabel('Vraies valeurs', fontsize=3)
    
    ## renommer les étiquettes 0 = pas de fuite, 1 = fuite
    ax.set_xticklabels(['non fuite', 'fuite'], fontsize=1.5)
    ax.set_yticklabels(['non fuite', 'fuite'], fontsize=1.5)
    #ax.set_title('Matrice de Confusion', fontsize=1)

    # Réduire la taille des étiquettes des axes
    ax.tick_params(axis='both', which='major', labelsize=4)
    
    #ax.set_ylim(len(cm)-0.5, -0.5)

    # Ajuster les marges pour rendre la figure plus compacte
    plt.tight_layout(pad=1.0)

    # Afficher la figure dans Streamlit
    st.subheader("Matrice de Confusion des données de test")
    st.pyplot(fig)
    return best_model, results[best_model], precision, recall, f1
    
st.subheader(' 1. Différence journalière')
acc_diff_journalier, result_diffs, precision_diffs, recall_diff, f1_diff = developpement_modele(diff_leaks)

st.subheader('2. Moyenne mobile')
acc_moyenne_mobile, result_moy_mobile, precision_moy, recall_moy, f1_moy = developpement_modele(moving_avg_leaks)
st.subheader('3 . Changements soudains')
acc_seuil_statique, result_changement_soudains, precision_changement, recall_changement, f1_changement = developpement_modele(changepoints)
st.subheader('4. Seuil statique')
acc_seuil_statique, result_statistique,precision_seuil, recall_seuil, f1_seuil = developpement_modele(static_leaks)
st.subheader('Comparaison des résultats des modèles de machine learning sur les données de test')



st.write(pd.DataFrame({
    'Méthode': ['Différence journalière', 'Seuil statique', 'Moyenne mobile', 'Changements soudains'],
    'Meilleur modèle': [acc_diff_journalier, acc_seuil_statique, acc_moyenne_mobile, acc_seuil_statique],
    'accuracy': [result_diffs, result_statistique, result_moy_mobile, result_changement_soudains],
    'precision': [precision_diffs, precision_seuil, precision_moy, precision_changement],
    'recall': [recall_diff, recall_seuil, recall_moy, recall_changement],
    'f1-score': [f1_diff, f1_seuil, f1_moy, f1_changement],
    'Debit entre': [heure, heure, heure, heure]
}))



### Tu peux me faire un texte pour expliquer les differents metriques : accuracy, precision, recall, f1-score
st.write("""
**Explication des métriques :**
- **Accuracy** : Le taux de prédictions correctes parmi toutes les prédictions. C'est une mesure globale de la performance du modèle.
- **Precision** : Le taux de prédictions correctes parmi les prédictions positives. Cela mesure la précision du modèle lorsqu'il prédit une classe positive.
- **Recall** : Le taux de prédictions correctes parmi les vraies valeurs positives. Cela mesure la sensibilité du modèle à détecter les vraies valeurs positives.
- **F1-score** : La moyenne harmonique de la précision et du recall. C'est une mesure globale de la précision et de la sensibilité du modèle.
""")




