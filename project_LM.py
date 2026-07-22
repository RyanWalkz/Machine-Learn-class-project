from datasets import load_dataset
import pandas as pd
import sys

ds = load_dataset("FronkonGames/steam-games-dataset")
df = ds["train"].to_pandas()

# Colunas que geralmente não ajudam na previsão numérica/categórica direta
cols_para_remover = [
    'peak_ccu', 'detailed_description', 'short_description', 
    'reviews', 'header_image', 'website', 'support_url', 
    'support_email', 'windows', 'mac', 'linux', 'movies', 'packages',
    'screenshots', 'metacritic_url', 'achievements', 'recommendations',
    'appID', 'name', 'metacritic_score', 'user_score', 'average_playtime_forever', 
    'average_playtime_2weeks', 'median_playtime_forever', 'median_playtime_2weeks']
df_clean = df.drop(columns=cols_para_remover)

print("Preços no dataframe:")
print(df_clean['price'].describe())
print(f"\nPrice range: ${df_clean['price'].min():.2f} to ${df_clean['price'].max():.2f}")
df_clean['price'].info()

df_clean['release_date'] = pd.to_datetime(df_clean['release_date'], errors='coerce') # Converter para datetime
df_clean['release_month'] = df_clean['release_date'].dt.month # Criar apenas release_month
df_clean['game_age'] = 2026 - df_clean['release_date'].dt.year # Criar game_age diretamente sem manter release_year
df_clean['game_age'] = df_clean['game_age'].fillna(df_clean['game_age'].median()) # Se houver datas inválidas, preencher game_age com a mediana
df_clean['estimated_owners'] = df_clean['estimated_owners'].str.replace(',', '') #remove as vírgulas para facilitar a conversão numérica
df_clean[['owners_min', 'owners_max']] = df_clean['estimated_owners'].str.split('-', expand=True) #divide a coluna de estimativa de proprietários em duas colunas: mínima e máxima
df_clean['owners_min'] = pd.to_numeric(df_clean['owners_min'], errors='coerce') #converte a coluna de proprietários mínimos para numérica, tratando erros
df_clean['owners_max'] = pd.to_numeric(df_clean['owners_max'], errors='coerce') #mesmo da coluna de proprietários minimos, mas para a máxima
df_clean['owners_mean'] = (df_clean['owners_min'] + df_clean['owners_max']) / 2 #Calcula o valor médio estimado de donos
df_clean['total_reviews'] = df_clean['positive'] + df_clean['negative'] #Cria o total de avaliações
df_clean['num_languages'] = df_clean['supported_languages'].apply(lambda x: len(x)) #Conta quantos idiomas o jogo suporta
df_clean['num_genres'] = df_clean['genres'].apply(lambda x: len(x)) #Conta quantos gêneros o jogo tem

cols_para_remover_final = [
    'release_date', 'estimated_owners','owners_min','owners_max','supported_languages',
    'genres','positive','negative','developers','categories','tags','notes',
    'full_audio_languages','score_rank','publishers','release_month','dlc_count','required_age']
df_model = df_clean.drop(columns=cols_para_remover_final)

def categorizar_preco_manual(preco):
    if preco == 0:
        return "gratuito"
    elif preco <= 2.5:
        return "muito_barato"
    elif preco <= 6:
        return "barato"
    elif preco <= 15:
        return "medio"
    else:
        return "caro"

df_model["precos_categorizados"] = df_model["price"].apply(categorizar_preco_manual)

df_model["precos_categorizados"].value_counts(normalize=True) * 100

bins = [
    -0.01,      # para incluir 0
    0,1.49,3.59,6.99,
    df_model["price"].max()]

labels = ["gratuito","baixo","medio_baixo","medio_alto","alto"]

df_model["precos_categorizados"] = pd.cut(
    df_model["price"],
    bins=bins,
    labels=labels)

df_model["precos_categorizados"].value_counts(normalize=True) * 100

X = df_model.drop(columns=["precos_categorizados"])
y = df_model["precos_categorizados"]
X = X.drop(columns=["price"], errors="ignore")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

model_final = RandomForestClassifier(
    max_depth=10,
    n_estimators=100,
    random_state=42,
    n_jobs=-1)

model_final.fit(X_train, y_train)

preds = model_final.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy final:", accuracy_score(y_test, preds))

import pandas as pd

importances = pd.Series(
    model_final.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importances)

df_model.to_csv("df_formatado.csv", index=False)

# Accuracy final: 0.40745066451872736