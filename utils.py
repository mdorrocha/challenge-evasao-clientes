from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier

SEED = 42

def executa_modelo(treino_x, teste_x, treino_y, teste_y, classificador, parametros_modelo={}):   
    """
    Esta função gera um modelo de machine learning e retorna métricas do mesmo:
    acurácia, precisão, recall e f1 score e matriz de confusão
    """ 
    # criando o modelo
    modelo = classificador(**parametros_modelo)
    # treinando o modelo
    modelo.fit(treino_x, treino_y)
    # efetuando previsões
    previsao_y = modelo.predict(teste_x)
    metricas = {
        modelo.__class__.__name__: {
            'Acurácia': accuracy_score(teste_y, previsao_y),
            'Precisão': precision_score(teste_y, previsao_y),
            'Recall': recall_score(teste_y, previsao_y),
            'F1 score': f1_score(teste_y, previsao_y)
        }
    }
    return metricas, previsao_y     

def gera_matriz_confusao(teste_y, previsao_y):
    print('Matriz de confusão')
    ConfusionMatrixDisplay.from_predictions(teste_y, previsao_y)

def otimiza_modelo(x, y):
    hiperparametros = {
        'n_estimators': list(range(50, 1001, 50)),
        'max_depth': list(range(5, 11)),
        'min_samples_split': list(range(2, 9)),
        'min_samples_leaf': list(range(1, 9)), 
        'criterion': ["gini", "entropy"],
        'bootstrap': [False, True]
    }
    scoring = ['accuracy', 'precision', 'recall']
    busca = RandomizedSearchCV(RandomForestClassifier(),
                    hiperparametros,
                    n_iter = 50,
                    scoring = scoring,
                    refit = 'recall',
                    cv = KFold(n_splits = 5, shuffle=True),
                    random_state = SEED)
    busca.fit(x, y)
    return busca
   
    
    


        
        
        
        
        
        
        
        
        
        
    
