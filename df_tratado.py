import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("df_formatado.csv")

X = df.drop(columns=['precos_categorizados'])
y = df['precos_categorizados']

model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
scores = cross_val_score(model, X, y, cv=5)

print(scores.mean())

# Resultado 01: 0.404024318931098