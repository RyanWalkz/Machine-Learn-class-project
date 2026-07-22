from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Carregar Dataset da Steam
print("Carregando dados da Steam...")
ds = load_dataset("FronkonGames/steam-games-dataset")
df_base = ds["train"].to_pandas()

# 2. Criar o DataFrame único do projeto filtrando o que não é preditivo direto
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

# 5. Métricas de Avaliações e Proporção Positivos
df_projeto['total_reviews'] = df_projeto['positive'] + df_projeto['negative']
df_projeto['positive_ratio'] = df_projeto['positive'] / df_projeto['total_reviews'].replace(0, 1)
df_projeto['positive_ratio'] = df_projeto['positive_ratio'].fillna(0.5)

# 6. Função Robusta para contagem de elementos
def contar_elementos_safe(x):
    if isinstance(x, (list, np.ndarray, tuple)):
        return len(x)
    if pd.isna(x):
        return 0
    if isinstance(x, str):
        return len([i for i in x.split(',') if i.strip()])
    return 0

# Criando as colunas que haviam sumido
df_projeto['num_languages'] = df_projeto['supported_languages'].apply(contar_elementos_safe)
df_projeto['num_genres'] = df_projeto['genres'].apply(contar_elementos_safe)

# 7. Extração de Gêneros Principais (One-Hot Encoding)
pop_genres = ['Indie', 'Action', 'Adventure', 'Casual', 'RPG', 'Strategy', 'Simulation']
for genre in pop_genres:
    df_projeto[f'genre_{genre}'] = df_projeto['genres'].astype(str).apply(lambda x: 1 if genre in x else 0)

# 8. Converter plataformas binárias para inteiros (0 ou 1)
for col in ['windows', 'mac', 'linux']:
    if col in df_projeto.columns:
        df_projeto[col] = df_projeto[col].astype(int)

# 9. Categorizar Preços (Seus Bins Originais)
bins = [-0.01, 0, 1.49, 3.59, 6.99, df_projeto["price"].max()]
labels = ["gratuito", "baixo", "medio_baixo", "medio_alto", "alto"]
df_projeto["precos_categorizados"] = pd.cut(df_projeto["price"], bins=bins, labels=labels)

# 10. Definição estrita das colunas de Features (X) e Target (y)
cols_features = [
    'game_age', 'owners_mean', 'total_reviews', 'positive_ratio', 
    'num_languages', 'num_genres', 'windows', 'mac', 'linux', 'dlc_count'
] + [f'genre_{g}' for g in pop_genres]

# Garantindo isolamento das variáveis do modelo
X = df_projeto[cols_features].fillna(0).copy()
y = df_projeto["precos_categorizados"].copy()

# 11. Divisão de Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 12. Configuração do Modelo
model_final = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

print("Treinando o modelo...")
model_final.fit(X_train, y_train)
preds = model_final.predict(X_test)

# 13. Exibir Resultados
print("\n=== RESULTADOS ===")
print("Acurácia final:", accuracy_score(y_test, preds))
print("\nRelatório de Classificação:\n", classification_report(y_test, preds))

# 14. Exportar base limpa
df_export = X.copy()
df_export["precos_categorizados"] = y
df_export.to_csv("df_formatado.csv", index=False)
print("\nArquivo 'df_formatado.csv' gerado e atualizado com sucesso!")