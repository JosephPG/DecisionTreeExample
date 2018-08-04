import matplotlib.pyplot as plt
import numpy


class ManagePlot(object):

    def __init__(self):
        pass

    def _caracteristicas_importante(self, col_feature, feature_name, arbol):
        caract = col_feature.shape[1] # Obtiene la cantidad de caracteristicas
        plt.barh(range(caract), arbol.feature_importances_)  # Genera barra
        plt.yticks(numpy.arange(caract), feature_name) # Propiedades de Y
        plt.xlabel('Importancia de caracteristicas')
        plt.ylabel('Caracteristica')
        plt.show()
