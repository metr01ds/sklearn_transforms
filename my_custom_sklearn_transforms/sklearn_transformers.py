from sklearn.base import BaseEstimator, TransformerMixin


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
        self.dados[self.columns] = self.dados[self.columns].where(self.dados[self.columns]<=self.val_max,other=self.valor, axis=1)
        return self.dados
