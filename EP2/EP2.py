"""
EP2 - MAP3122

File
-------
EP2.py

Authors
-------
    Joao Victor Texeira Degelo      - 11803479
    Pedro Henrique Martins de Santi - 11803483
"""

import numpy as np
import matplotlib.pyplot as plt

def V(S, t, K, T):
    if (t == T):
        if ((S - K) > 0):
            # print("S = ", S, "\n")
            # print("K = ", K, "\n")

            return add_float(S, -K)
        else:
            return 0
    elif (S == 0 and t >= 0):
        return 0
    elif (S == "+inf"):
        return "+inf"

def Lucro(n, S, K, T, Vpremio):
    # Vf = V(S, T, K, T)
    # print("Vf = ", Vf, "\n")
    # print("n = ", n, "\n")
    # print("Vp = ", Vp, "\n")
    # print("n*Vf", n*Vf, "\n")

    lucro = (n*V(S, T, K, T)) - Vpremio
    return lucro
    # if ((S - K) > 0):
    #     return (S - K) - Vp
    # else:
    #     return 0 - Vp

# calor
# def x(S, t, K, T, r, sigma):
#     return np.log(S/K) + ((r - ((sigma**2)/2))*(T-t))

# def S(S, t, K, T, r, sigma):
#     xf = x(S, t, K, T, r, sigma)
#     return K*((np.e)**(xf-((r - ((sigma**2)/2))*(T-t))))

# def u(S, t, K, T, r, sigma):
#     V()

#

'''

Vc (S, t) = S N(d1) - K e^(-r(T - t)) N(-d2)

tal(t) = T - t

u ( x, 0  ) = K * max (e^x - 1, 0)
u (-L, tal) = 0
u ( L, tal) = k * e^(L + (sigma**2) * tal/2)

L grande:
u ( L, tal) = S

delta x = 2L/N
delta tal = T/M

'''

def Vc():
    print()

def Vp():
    print()

def u(j, i, K, sigma, L, N, T, M):
    if (j == 0):
        if ((((np.e)**x_i(i, L, N)) - 1) > 0):
            # print(i, j, "max\n")

            return (((np.e)**x_i(i, L, N)) - 1)
        else:
            # print(i, j, "zero max\n")

            return 0

    elif (i == 0):
        # print(i, j, "zeru\n")

        return 0

    elif (i == N):
        # print(i, j, "semi\n")
        return K*(np.e**(L + (sigma**2)*(tal_j(j, T, M)/2)))

    else:
        # print(i, j, "Calculando u_ij\n")
        u_ij = u(j - 1, i, K, sigma, L, N, T, M)

        # print(i, j, "Calculando u_i-1\n")
        u_ip = u(j - 1, i - 1, K, sigma, L, N, T, M)

        # print(i, j, "Calculando u_i+1\n")
        u_in = u(j - 1, i + 1, K, sigma, L, N, T, M)

        # print(i, j, "conta louca\n")
        return u_ij + ((2*L*(M**2)*(sigma**2))/(N*(T**2)*2))*(u_ip - 2*u_ij + u_in)

def x_i(i, L, N):
    return ((i*(2*L/N))-L)

def tal_j(j, T, M):
    return (j*(T/M))

def S_ij(j, i, K, r, sigma, L, N, T, M):
    return K*((np.e)**(x_i(i, L, N)-((r - ((sigma**2)/2))*tal_j(j, T, M))))

def V_ij(j, i, K, r, sigma, L, N, T, M):
    return u(j, i, K, sigma, L, N, T, M)*(np.e)**(-r*tal_j(j, T, M))




# Método para evitar erro por ponto flutuante
def add_float(a, b):
    cont = 0
    while ((a - int(a)) != 0 or (b - int(b)) != 0):
        a *= 10
        b *= 10
        cont += 1
    dif = a + b
    pot = 10**cont
    return (dif/pot)

def main():

    K = 1
    sigma = 1/100
    T = 1
    r = 1/100
    N = 10000
    L = 10
    M = 10000

    i = 20
    j = 20

    print(u(j, i, K, sigma, L, N, T, M))

    # print("\n-----------------------------------------------------\n")
    # print("Permissão comprar 1 mi dolares. Dolar a 5,20(K) daqui 3 meses(T)\n")
    # print("Precipicacao da opcao: permio (Vp) = 40.000\n")
    # print("Taxa de cambio na hora da venda ou final de T (S)")

    # n = 1000000
    # K = 5.20
    # T = 3
    # Vpremio = 40000

    # S = 5.05
    # while (S <= 5.45):
    #     print("para S = ", S, ":\n")
    #     lucro = Lucro(n, S, K, T, Vpremio)
    #     print(lucro, "\n")
    #     S = add_float(S, 0.05)



# Função main
if __name__ == "__main__":
    try:
        main()
    except     KeyboardInterrupt:
        print("deu erro:  :(\n BETTER luck next time")
