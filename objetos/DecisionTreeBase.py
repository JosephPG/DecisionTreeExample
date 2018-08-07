from sklearn.tree import DecisionTreeClassifier


class DecisionTreeBase(object):

    def __init__(self, max_depth=None, *args, **kwargs):
        super(DecisionTreeBase, self).__init__(*args, **kwargs)
        self._max_depth = max_depth
        self._arbol = DecisionTreeClassifier(max_depth=max_depth)
        self._feature_name = []
        self._target_name = []
        self._x_train = None
        self._x_test = None
        self._y_train = None
        self._y_test = None

    def _entrenar_arbol(self):
        self._arbol.fit(self._x_train, self._y_train)

    def imp_score_predict(self):
        """Porcentaje que asegura cuan seguro puede ser
        acertar la prediccion"""
        train_score_n = self._arbol.score(self._x_train, self._y_train)
        test_score_n = self._arbol.score(self._x_test, self._y_test)
        print("\n ======================")
        print('Train Score: {}'.format(train_score_n))
        print('Test Score: {}'.format(test_score_n))
        print("====================== \n")

    def imp_predict(self):
        """Imprime el resultado de la clasificacion"""
        predict_result = self._arbol.predict(self._x_test)
        print("\n ======================")
        print('Prediccion: {}'.format(predict_result))
        print("======================\n")
