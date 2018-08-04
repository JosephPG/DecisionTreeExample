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
        arbol.entrenar_arbol()
        arbol.imprimir_resultados()
        arbol.generar_dot()
        arbol.leer_dot()
        arbol.barra_feature_important()


if __name__ == '__main__':
    DecisionTreeExample.main()
