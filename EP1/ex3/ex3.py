"""
Exercicio 3 - EP1 - MAP3122

File
-------
ex3.py

Authors
-------
    Joao Victor Texeira Degelo      - 11803479
    Pedro Henrique Martins de Santi - 11803483
"""

import numpy as np
import matplotlib.pyplot as plt

def autovalor_certo(A):
    autovalores_certo, autovetores_certo = np.linalg.eig(A)
    print("autovalores corretos: \n", autovalores_certo, "\n")
    print("autovetores corretos: \n", autovetores_certo, "\n")

def autovalor(A, k):
    """
    Funcao que calcula os autovalores e autovetores de uma matriz usando a fatoracao QR
    e imprime a matriz resultante de k interacoes

    Parameters
    -------------
    A : np.array
        Matriz da qual se quer achar os autovalores
    k : int
        Numero de iteracaoes

    Returns
    -------------
    lambda : np.array
        Array com os autovalores da matriz
    V : np.array
        Array com os autovetores da matriz
    """
    i = 0
    while (i != k):
        Q, R = fatoracao_QR(A)
        A = R @ Q

        if(i == 0):
            V = Q
        else:
            V = V @ Q

        i = i + 1
    print("Matriz apos", k, "fatoracoes:\n", A, "\n")
    return np.diag(A), V

def fatoracao_QR(A):
    """
    Funcao que realiza a fatoracao QR

    Parameters
    -------------
    A : np.array
        Matriz que sera fatorada

    Returns
    -------------
    Q : np.array
        Matriz Q da fatoracao QR
    R : np.array
        Matriz R da fatoracao QR
    """
    n = A.shape[0] # dimensao da matrix nxn
    I = np.eye(n) # matriz identidade

    i = 1
    while (i < n):

        Al = np.tril(A)
        ai = Al[:, i - 1] # coluna i
        ei = I[:, i - 1] # vetor da base canonica
        gama = np.sign(A[:, i - 1][i - 1]) # sinal do elemento mii de ai
        vi = np.array([(ai + gama*np.linalg.norm(ai)*ei)]).T

        Hi = I - (2/(vi.T @ vi))*(vi @ vi.T) # Transformacao de Householder

        if (i == 1): # atualiza valor total de Q
            Q = Hi
        else:
            Q = Hi @ Q

        Ri = Hi @ A
        A = Ri
        i = i + 1

    return Q, Ri

def main():
    # valor exato autovalores de A pelo polinomio caracteristico

    # valor exato dos autovetores de A resolvendo sitemas lineares (A - lambda_k*I)x = 0, k = 1, 2, 3
    # det (A - lambda_k*I) = 0, x!=0, duas equacoes indep. inifinitas solucoes para x
    # Solucao unica: normalizacao e sinal positivo para a primeira entrada do autovetor
    # autovetores v_k devem ser normalizados: ||v_k|| = 1

    # verificar convergencia

    # Aprox numerica dos autovalores e autovetores de A por QR

    A1 = np.array([[6, -2, -1], [-2, 6,-1], [-1, -1, 5]])
    print("Para o vetor:\n", A1, "\n")

    lambda1, V1 = autovalor(A1, 1000)
    print("autovalores encontrados: \n", lambda1, "\n")
    print("autovetores encontrados: \n", V1, "\n")
    autovalor_certo(A1)

    print("\n-----------------------------------------------------\n")
    # valor autovalores de A pelo polinomio caracteristico
    # valor autovalores de A por QR

    A2 = np.array([[1, 1], [-3, 1]])
    print("Para o vetor:\n", A2, "\n")

    lambda2, V2 = autovalor(A2, 1000)
    print("autovalores encontrados: \n", lambda2, "\n")
    print("autovetores encontrados: \n", V2, "\n")
    autovalor_certo(A2)

    print("\n-----------------------------------------------------\n")
    # valor autovalores de A pelo polinomio caracteristico
    # valor autovalores de A por QR

    A3 = np.array([[3, -3], [0.33333, 5]])
    print("Para o vetor:\n", A3, "\n")

    lambda3, V3 = autovalor(A3, 1000)
    print("autovalores encontrados: \n", lambda3, "\n")
    print("autovetores encontrados: \n", V3, "\n")
    autovalor_certo(A3)


    # Aplicar metodo QR as duas matrizes B do exercicio 1


# Função main
if __name__ == "__main__":
    try:
        main()
    except     KeyboardInterrupt:
        print("deu erro:  :(\n BETTER luck next time")
