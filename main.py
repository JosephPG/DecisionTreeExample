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
        arbol = DecisionTreeFirst()
        arbol.imp_score_predict()
        arbol.imp_predict()
        arbol.abrir_dot()
        arbol.graficar_caracteristicas_importantes()


if __name__ == '__main__':
    DecisionTreeExample.main()
