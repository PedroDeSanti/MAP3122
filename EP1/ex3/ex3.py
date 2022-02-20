"""
Exercicio 3 - EP1 - MAP3122

File
-------
ex1.py

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

def normaliza_autovetor(V):
    """
    Funcao que normaliza os autovetores dada uma matriz na
    qual suas colunas sao formadas por autovetores

    Parameters
    -------------
    V : np.array
        Matriz com os autovetores a serem normalizados

    Returns
    -------------
    V : np.array
        Matriz com os autovetores normalizados
    """
    n = V.shape[0] # dimensao da matrix nxn
    for i in range(1, n + 1):
        V[:, i - 1] = V[:, i - 1]/np.linalg.norm(V[:, i - 1])
    return V

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
    V = normaliza_autovetor(V)
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
    print("\n-----------------------------------------------------\n")
    # Aprox numerica dos autovalores e autovetores de A por QR

    A1 = np.array([[6, -2, -1], [-2, 6,-1], [-1, -1, 5]])
    print("Para o vetor:\n", A1, "\n")

    lambda1, V1 = autovalor(A1, 1000)
    print("autovalores encontrados: \n", lambda1, "\n")
    print("autovetores encontrados: \n", V1, "\n")
    autovalor_certo(A1)


    print("\n-----------------------------------------------------\n")
    # valor autovalores de A por QR

    A2 = np.array([[1, 1], [-3, 1]])
    print("Para o vetor:\n", A2, "\n")

    lambda2, V2 = autovalor(A2, 1000)
    print("autovalores encontrados: \n", lambda2, "\n")
    print("autovetores encontrados: \n", V2, "\n")
    autovalor_certo(A2)


    print("\n-----------------------------------------------------\n")
    # valor autovalores de A por QR

    A3 = np.array([[3, -3], [0.33333, 5]])
    print("Para o vetor:\n", A3, "\n")

    lambda3, V3 = autovalor(A3, 1000)
    print("autovalores encontrados: \n", lambda3, "\n")
    print("autovetores encontrados: \n", V3, "\n")
    autovalor_certo(A3)

    print("\n-----------------------------------------------------\n")
    # Aplicar metodo QR as matrizes B do exercicio 1

    A4_maior_autovalor = 10.408955841652919
    A4 = np.array([[0.17051463, 1.6237461 , 1.38462038, 1.22281179, 1.23890696,
        1.65630898, 1.08484005, 0.61856201, 1.19625238, 1.0558061 ],
       [1.6237461 , 1.54646513, 1.13958464, 0.63536891, 1.42838108,
        0.51685674, 0.5150195 , 0.92449135, 0.86321038, 0.94429435],
       [1.38462038, 1.13958464, 0.9489933 , 0.60925617, 0.71583598,
        0.80221421, 0.60251725, 1.17650382, 0.6701969 , 1.4838219 ],
       [1.22281179, 0.63536891, 0.60925617, 1.21483832, 0.83670213,
        0.77076574, 1.10084781, 1.35923738, 1.10276765, 1.10306458],
       [1.23890696, 1.42838108, 0.71583598, 0.83670213, 0.98313316,
        1.39073056, 1.60261264, 0.43617306, 1.3141646 , 0.57046673],
       [1.65630898, 0.51685674, 0.80221421, 0.77076574, 1.39073056,
        0.61878981, 1.09173683, 1.25251016, 0.25498298, 1.07592248],
       [1.08484005, 0.5150195 , 0.60251725, 1.10084781, 1.60261264,
        1.09173683, 0.6237405 , 1.07471313, 1.05053754, 1.30109561],
       [0.61856201, 0.92449135, 1.17650382, 1.35923738, 0.43617306,
        1.25251016, 1.07471313, 1.32856511, 1.52231558, 1.29051216],
       [1.19625238, 0.86321038, 0.6701969 , 1.10276765, 1.3141646 ,
        0.25498298, 1.05053754, 1.52231558, 0.79196338, 1.15819024],
       [1.0558061 , 0.94429435, 1.4838219 , 1.10306458, 0.57046673,
        1.07592248, 1.30109561, 1.29051216, 1.15819024, 1.84824888]])
    lambda4, V4 = autovalor(A4, 1000)
    print("autovalores encontrados por QR: \n", lambda4, "\n")
    print("autovalor maximo por metodo das potencias : \n", A4_maior_autovalor, "\n")

    print("\n-----------------------------------------------------\n")
    # Aplicar metodo QR as matrizes B do exercicio 1

    A5_maior_autovalor = 14.859966431658659
    A5 = np.array([[  7.04118903, -19.7155082 ,  -7.93999503,  -6.40677919,
         -4.00401075,  37.65560326,  35.19625493,  25.11640474,
        -37.07525968, -22.4459207 ],
       [-12.37802969, -33.27160344, -20.39189355,  -3.14943047,
          3.30137442,  77.26784942,  63.14760573,  42.90352228,
        -77.52552898, -28.08964194],
       [ -3.21864901,  -3.38241489,  12.34632419,  -4.61206869,
         -1.88392124,   6.3582982 ,  11.6223586 ,   7.73156013,
         -7.88500526, -10.25418829],
       [-10.14109892, -25.34951688, -10.7550899 ,   6.80954343,
          0.69810241,  43.95729956,  39.85016672,  26.74350575,
        -45.9373954 , -17.92621349],
       [  1.63168393,  -0.34156456,  -0.69897453,  -4.30000729,
          8.87721265,   9.51489737,   5.53659499,   4.21078032,
        -10.23458921,  -5.06987832],
       [ -7.50623629, -13.53936834,  -4.25590396,  -5.17273638,
         -0.25786194,  33.39951028,  24.52627178,  15.23117591,
        -21.63317927, -13.69312593],
       [ -8.5490966 , -43.54871671, -19.44465166,  -5.13792842,
         -3.31546842,  80.09560004,  75.89380067,  49.32873241,
        -79.57551041, -35.17618458],
       [ -2.32415504,  -8.51541136,  -6.36968257,  -3.12829849,
          3.6831618 ,  20.07555807,  16.13382338,  16.68876978,
        -22.12150482,  -4.99729809],
       [ -5.84878782, -20.28576221, -10.04447414,  -5.35665937,
         -0.42547524,  40.11093012,  33.58455114,  22.0754673 ,
        -28.71714297, -16.8571621 ],
       [ -6.65578683, -19.76182166,  -6.94455928,  -4.78497363,
         -2.75824808,  33.2349799 ,  33.22086863,  24.30099283,
        -32.73180763,  -9.44760363]])
    lambda5, V5 = autovalor(A5, 1000)
    print("autovalores encontrados por QR: \n", lambda5, "\n")
    print("autovalor maximo por metodo das potencias : \n", A5_maior_autovalor, "\n")

    print("\n-----------------------------------------------------\n")
    # Aplicar metodo QR as matrizes B do exercicio 1

    A6_maior_autovalor = 18.750000002690648
    A6 = np.array([[ 17.74988646, -12.92999882,  -2.98476039,  -5.51148418,
         -8.34485949,  26.83216589,  22.89341007,  16.41267457,
        -20.19578586, -23.40960939],
       [ 24.14492168,  43.74369189,  25.18514103,  -6.61824559,
         -9.63465139, -60.63820674, -48.72497758, -28.22746739,
         61.25393847,   5.38041532],
       [  4.18183369,   4.91235279,  15.69237873,  -1.4280803 ,
         -3.91686345, -10.79721423,  -7.61300396,  -1.51074603,
         10.67625126,   0.4001667 ],
       [ 16.17204874,  30.51044552,  19.86137198,   7.21438093,
         -4.52420591, -56.20542516, -46.44305275, -25.26405857,
         53.95156508,  13.34965293],
       [  4.68227466,  -2.81201604,   2.15874158,  -1.10225547,
          6.15598206,   3.68209687,   4.85126276,   1.6046517 ,
         -0.33288728,  -9.94663698],
       [ 10.70469824,  17.17503088,  12.66382528,  -2.15224703,
         -5.48075322, -22.23149339, -26.36907393, -12.71500246,
         32.21066169,   5.33619517],
       [ 18.91800545,  12.61912856,  12.92975032,  -8.5902199 ,
         -8.7362553 , -21.19507483,  -5.24490554,  -8.77995949,
         24.67333689,  -8.68822846],
       [  7.75104148,   2.21117553,   6.7949665 ,  -2.55530662,
         -8.16791758,  -4.45040284,  -2.16771244,  10.38084635,
          9.56389892, -10.75708684],
       [ 10.94648834,   8.76686644,   9.04477127,  -3.71581084,
         -6.67434715, -16.38909344, -12.34161242,  -5.52265908,
         29.35059282,  -4.42278414],
       [  7.82272313,   1.7601541 ,   6.09036992,  -4.46183203,
         -7.55858073,  -3.19245308,  -1.16390968,   1.53877282,
          7.44210472,   1.14863968]])
    lambda6, V6 = autovalor(A6, 1000)
    print("autovalores encontrados por QR: \n", lambda6, "\n")
    print("autovalor maximo por metodo das potencias : \n", A6_maior_autovalor, "\n")

# Função main
if __name__ == "__main__":
    try:
        main()
    except     KeyboardInterrupt:
        print("deu erro:  :(\n BETTER luck next time")
