from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    

#localiza valores maiores que 10 na coluna especificada e substitui por "10"
class CorrigeNotas(BaseEstimator, TransformerMixin):
    def __init__(self, dados, columns, val_max, valor):
        self.dados = dados
        self.columns = columns
        self.val_max = val_max
        self.valor = valor

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        data[self.columns] = self.dados[self.columns].where(self.dados[self.columns]<=self.val_max,other=self.valor, axis=1)
        return data
    
#Insere coluna com médias
class insereMedias(BaseEstimator, TransformerMixin):
    def __init__(self, dados, columns):
        self.dados = dados
        self.columns = columns
        

    def fit(self, X, y=None):
        return self
        
    
    def transform(self, X):
        data = X.copy()
        data['MEDIA'] = data[self.columns].mean(axis=1)
        return data

#Insere coluna contendo a diferenças entre as médias gerais de cada materia e a média daquele aluno
class insereDifMedias(BaseEstimator, TransformerMixin):
    def __init__(self, dados, column1, column2, column3, column4):
        self.dados = dados
        self.column1 = column1
        self.column2 = column2
        self.column3 = column3
        self.column4 = column4
        

    def fit(self, X, y=None):
        return self
        
    
    def transform(self, X):
        data = X.copy()
        data[column1] = (sum(data[column1])/len(data[column1])) - data[column1].mean(axis=1)
        data[column2] = (sum(data[column2])/len(data[column2])) - data[column2].mean(axis=1)
        data[column3] = (sum(data[column3])/len(data[column3])) - data[column3].mean(axis=1)
        data[column4] = (sum(data[column4])/len(data[column4])) - data[column4].mean(axis=1)
        return data
    
#executa o SMOTE
class ExecutaSmote(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        print(self)
        
        X2 = X.copy()
        y2 = y.copy()
        
        X, y = SMOTE().fit_sample(X2, y2)
        ret = (X, y)
        return ret

    def transform(self, X):
        return self   
