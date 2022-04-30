import pandas as pd
from caracteristicasDerivadas import H_entropy, proporcionVocalesConsonantes, posicionPrimerDigito

from sklearn import metrics, model_selection, tree

#Carga de datos
df = pd.read_csv("modelo_compara\dga_data_small.csv")

#Eliminacion de caracteristicas irrelevante o repetidas
df.drop(['host','subclass'], axis=1, inplace=True)

#Codificacion de variable objetivo
df['isDGA'] = df['isDGA'].replace(to_replace='dga', value=1)
df['isDGA'] = df['isDGA'].replace(to_replace='legit', value=0)


df['longitud'] = df['domain'].str.len()
df['digitos'] = df['domain'].str.count('[0-9]')
df['entropia'] = df['domain'].apply(H_entropy)
df['proporcionVocalesConsonantes'] = df['domain'].apply(proporcionVocalesConsonantes)
df['posicionPrimerDigito'] = df['domain'].apply(posicionPrimerDigito)

df.drop(['domain'], axis=1, inplace=True)


print('Final features:', df.columns)
print(df.head())

df_final = df


target = df_final['isDGA']
feature_matrix = df_final.drop(['isDGA'], axis=1)

print('Final features:', feature_matrix.columns)
feature_matrix.head()

feature_matrix_train, feature_matrix_test, target_train, target_test = model_selection.train_test_split(feature_matrix, target, test_size=0.25, random_state=31)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature_matrix_train, target_train)

from joblib import dump
dump(clf, 'my_model.joblib')
print("hola")

print(feature_matrix_train.count())

print(feature_matrix_test.count())

#Metricas

target_pred = clf.predict(feature_matrix_test)

print(metrics.accuracy_score(target_test, target_pred))
print('Matriz de confusion /n',metrics.confusion_matrix(target_test, target_pred))
print(metrics.classification_report(target_test, target_pred, target_names=['legit', 'dga']))