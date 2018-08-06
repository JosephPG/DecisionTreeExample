import numpy
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from .DecisionTreeBase import DecisionTreeBase


class ManagePlot(DecisionTreeBase):

    def __init__(self, *args, **kwargs):
        super(ManagePlot, self).__init__(*args, **kwargs)

    def graficar_caracteristicas_importantes(self):
        cant_feature = self._x_train.shape[1]
        plt.barh(range(cant_feature), self._arbol.feature_importances_)  # Genera barra
        plt.yticks(numpy.arange(cant_feature), self._feature_name) # Propiedades de Y
        plt.xlabel('Importancia de caracteristicas')
        plt.ylabel('Caracteristica')
        plt.show()

    def graficar_clasificacion(self):
        distance_step = 0.02
        cant_result = len(self._target_name)
        color_graph = "rby"  #Cada letra representa un color 
        # Se itera el numero de combinacion en pares para cada columna de caracteristica
        for pair_num, pair_column in enumerate([[0,1], [0,2], [0,3],
                                                [1,2], [1,3], [2,3]]):
            x = self._x_train[:, pair_column] # Obtiene todas las filas, columnas de la iteracion actual
            y = self._y_train
            clf = DecisionTreeClassifier().fit(x, y)
            # Genera un agrupado de 6 graficos con 2 filas y 3 columnas
            # y el ultimo parametro indica la posicion del grafico actual en el agrupado
            plt.subplot(2, 3, pair_num+1)

            # Obtiene el valor minimo y maximo de cada columna
            x_min, x_max = x[:, 0].min()-1, x[:, 0].max()+1
            y_min, y_max = x[:, 1].min()-1, x[:, 1].max()+1

            # Generar rango desde el minimo al maximo de 0.02
            x_range = numpy.arange(x_min, x_max, distance_step)
            y_range = numpy.arange(y_min, y_max, distance_step)

            # Genera matriz de coordenada para el plano del grafico
            xx, yy = numpy.meshgrid(x_range, y_range)

            # convierte xx y yy en listas de 1 dimension, y se concatena en dos columnas
            z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
            z = z.reshape(xx.shape) # convierte el array en las mismas dimensiones que xx

            # Se llena el contenido del grafico, y se indica la paleta de colores
            # Paired pertence al grupo cualitativo 
            content_data = plt.contour(xx, yy, z, cmap=plt.cm.Paired)

            plt.xlabel(self._feature_name[pair_column[0]])
            plt.ylabel(self._feature_name[pair_column[1]])
            plt.axis('tight')

            # Recorre la cantidad de targets para graficar cada resultado predecido
            for target_val, color in zip(range(cant_result), color_graph):
                array_position = numpy.where(y == target_val) # Obtiene array de posicion del target actual
                # Scatter es un tipo de grafico que muestra la data como coleccion de puntos
                plt.scatter(x[array_position, 0], x[array_position, 1],
                            label=self._target_name[target_val], cmap=plt.cm.Paired)

            plt.axis('tight')

        plt.suptitle("Clasificador de arbol: \n Las lineas representan que hay sobre ajuste")
        plt.legend()
        plt.show()

