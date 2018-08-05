import numpy
import matplotlib.pyplot as plt
from .DecisionTreeBase import DecisionTreeBase


class ManagePlot(DecisionTreeBase):

    def __init__(self, *args, **kwargs):
        super(ManagePlot, self).__init__(*args, **kwargs)

    def _caracteristicas_importante(self, cant_feature):
        plt.barh(range(cant_feature), self._arbol.feature_importances_)  # Genera barra
        plt.yticks(numpy.arange(cant_feature), self._feature_name) # Propiedades de Y
        plt.xlabel('Importancia de caracteristicas')
        plt.ylabel('Caracteristica')
        plt.show()
