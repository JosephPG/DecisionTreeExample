from objetos.DecisionTreeFirst import DecisionTreeFirst
from objetos.DecisionTreeExampleZero import DecisionTreeExampleZero


class DecisionTreeExample:

    @staticmethod
    def main(*args, **kwargs):
        #DecisionTreeExample.example_zero()
        DecisionTreeExample.example_first()

    @staticmethod
    def example_zero():
        arbol = DecisionTreeExampleZero()
        arbol.feature_for_setosa()
        arbol.print_data_col()

    @staticmethod
    def example_first():
        arbol = DecisionTreeFirst(max_depth=3)
        arbol.imp_score_predict()
        arbol.imp_predict()
        arbol.abrir_dot()
        #arbol.graficar_caracteristicas_importantes()
        arbol.graficar_clasificacion()

if __name__ == '__main__':
    DecisionTreeExample.main()
