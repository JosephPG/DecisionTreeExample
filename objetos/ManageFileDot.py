import graphviz
from sklearn.tree import export_graphviz


class ManageFileDot(object):

    def __init__(self, file_name):
        self.__root_path = 'docs/'
        self._file_name = file_name

    def _crear_dot(self, arbol, target_names, features_names,
                   impurity=False, filled=True):
        """Exportar archivo dot que contiene las estructura del arbol"""
        try:
            path_file = self.__root_path + self._file_name
            export_graphviz(arbol, out_file=path_file,
                            class_names=target_names,
                            feature_names=features_names,
                            impurity=impurity,
                            filled=filled)
        except Exception as e:
            print('Error in export graph dot:' + str(e))

    def leer_dot(self):
        """Abre el dot y genera un pdf"""
        try:
            path_file = self.__root_path + self._file_name
            graph_file = self.__root_path + \
                         self._file_name.replace('.', '-')
            with open(path_file) as f:
                dot_file = f.read()
            graf = graphviz.Source(dot_file, directory=self.__root_path)
            graf.view()
        except (FileNotFoundError, Exception) as e:
           print('Error en graph')
