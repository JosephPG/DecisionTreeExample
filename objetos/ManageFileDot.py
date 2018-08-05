import graphviz
from sklearn.tree import export_graphviz
from .DecisionTreeBase import DecisionTreeBase

class ManageFileDot(DecisionTreeBase):

    def __init__(self, file_name='', *args, **kwargs):
        super(ManageFileDot, self).__init__(*args, **kwargs)
        self._root_path = 'docs/'
        self._file_name = file_name
        self._full_path_file = self._root_path + self._file_name

    def _crear_dot(self, impurity=False, filled=True):
        """Exportar archivo dot que contiene las estructura del arbol"""
        try:
            export_graphviz(self._arbol, out_file=self._full_path_file,
                            class_names=self._target_name,
                            feature_names=self._feature_name,
                            impurity=impurity,
                            filled=filled)
        except Exception as e:
            print('Error in export graph dot:' + str(e))

    def _leer_dot(self):
        """Abre el dot y genera un pdf"""
        try:
            with open(self._full_path_file) as f:
                dot_file = f.read()
            graf = graphviz.Source(dot_file, directory=self._root_path)
            graf.view()
        except (FileNotFoundError, Exception) as e:
           print('Error en graph')
