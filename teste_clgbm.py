from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

# 1. Carregar Dataset da Steam
print("Carregando dados da Steam...")
ds = load_dataset("FronkonGames/steam-games-dataset")
df_base = ds["train"].to_pandas()

# 2. Remover colunas sem valor preditivo direto
cols_inuteis = [
    'peak_ccu', 'detailed_description', 'short_description', 
    'reviews', 'header_image', 'website', 'support_url', 
    'support_email', 'movies', 'packages',
    'screenshots', 'metacritic_url', 'achievements', 'recommendations',
    'appID', 'name', 'user_score', 'average_playtime_forever', 
    'average_playtime_2weeks', 'median_playtime_forever', 'median_playtime_2weeks'
]
df_projeto = df_base.drop(columns=cols_inuteis, errors='ignore').copy()

# 3. Tratamento de Datas e Idade do Jogo (Base em 2026)
df_projeto['release_date'] = pd.to_datetime(df_projeto['release_date'], errors='coerce')
df_projeto['game_age'] = 2026 - df_projeto['release_date'].dt.year
df_projeto['game_age'] = df_projeto['game_age'].fillna(df_projeto['game_age'].median())

# 4. Tratamento dos Proprietários (Owners)
df_projeto['estimated_owners'] = df_projeto['estimated_owners'].str.replace(',', '')
df_projeto[['owners_min', 'owners_max']] = df_projeto['estimated_owners'].str.split('-', expand=True)
df_projeto['owners_min'] = pd.to_numeric(df_projeto['owners_min'], errors='coerce')
df_projeto['owners_max'] = pd.to_numeric(df_projeto['owners_max'], errors='coerce')
df_projeto['owners_mean'] = (df_projeto['owners_min'] + df_projeto['owners_max']) / 2
df_projeto['owners_mean'] = df_projeto['owners_mean'].fillna(df_projeto['owners_mean'].median())

# 5. Métricas de Avaliações
df_projeto['total_reviews'] = df_projeto['positive'] + df_projeto['negative']
df_projeto['positive_ratio'] = df_projeto['positive'] / df_projeto['total_reviews'].replace(0, 1)
df_projeto['positive_ratio'] = df_projeto['positive_ratio'].fillna(0.5)

# 6. Função para contagem de elementos
def contar_elementos_safe(x):
    if isinstance(x, (list, np.ndarray, tuple)):
        return len(x)
    if pd.isna(x):
        return 0
    if isinstance(x, str):
        return len([i for i in x.split(',') if i.strip()])
    return 0

df_projeto['num_languages'] = df_projeto['supported_languages'].apply(contar_elementos_safe)
df_projeto['num_genres'] = df_projeto['genres'].apply(contar_elementos_safe)

# 7. Suas Variáveis de Feature Engineering (Porte do Jogo)
def definir_porte(genres):
    genres_str = str(genres)
    if 'Indie' in genres_str:
        return 'Indie'
    elif any(g in genres_str for g in ['Action', 'Adventure', 'RPG', 'Strategy']):
        return 'AAA_Core'
    elif 'Casual' in genres_str or 'Simulation' in genres_str:
        return 'Casual'
    else:
        return 'Outro'

df_projeto['porte_jogo'] = df_projeto['genres'].apply(definir_porte)
df_projeto['is_indie'] = (df_projeto['porte_jogo'] == 'Indie').astype(int)
df_projeto['is_aaa'] = (df_projeto['porte_jogo'] == 'AAA_Core').astype(int)
df_projeto['is_casual'] = (df_projeto['porte_jogo'] == 'Casual').astype(int)

# 8. Plataformas
for col in ['windows', 'mac', 'linux']:
    if col in df_projeto.columns:
        df_projeto[col] = df_projeto[col].astype(int)

# 9. Categorizar Preços
bins = [-0.01, 0, 1.49, 3.59, 6.99, df_projeto["price"].max()]
labels = ["gratuito", "baixo", "medio_baixo", "medio_alto", "alto"]
df_projeto["precos_categorizados"] = pd.cut(df_projeto["price"], bins=bins, labels=labels)

# 10. Seleção de Features
cols_features = [
    'game_age', 'owners_mean', 'total_reviews', 'positive_ratio', 
    'num_languages', 'num_genres', 'windows', 'mac', 'linux', 'dlc_count',
    'is_indie', 'is_aaa', 'is_casual'
]

X = df_projeto[cols_features].fillna(0).copy()
y = df_projeto["precos_categorizados"].astype(str)

# Mapeando os textos do alvo (y) para números para compatibilidade total
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 11. Divisão de Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 12. Treinamento com LightGBM
model_lgbm = LGBMClassifier(
    n_estimators=300,        # Mais árvores sequenciais
    learning_rate=0.03,      # Taxa de aprendizado suave para evitar overfitting
    max_depth=7,             # Profundidade moderada
    num_leaves=31,           # Número de folhas padrão do LightGBM
    subsample=0.8,           # Amostragem de linhas
    colsample_bytree=0.8,    # Amostragem de colunas por árvore
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("Treinando o modelo LightGBM...")
model_lgbm.fit(X_train, y_train)

# 13. Previsões e Reversão dos Rótulos
preds_encoded = model_lgbm.predict(X_test)
preds = label_encoder.inverse_transform(preds_encoded)
y_test_original = label_encoder.inverse_transform(y_test)

# 14. Exibição dos Resultados
print("\n=== RESULTADOS (LIGHTGBM) ===")
print("Acurácia final:", accuracy_score(y_test_original, preds))
print("\nRelatório de Classificação:\n", classification_report(y_test_original, preds))