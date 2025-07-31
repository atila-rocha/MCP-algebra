#from fastmcp import FastMCP
from mcp.server.fastmcp import FastMCP
import numpy as np
import sys
from math import pow

app=FastMCP('operadores de matrizes')

@app.tool(name='ping')
def ping():
    '''Caso o usuário escrever ping, retornará pong, isso significará ao usuário que o servidor MCP está funcionando e retornando o que foi pedido'''
    return 'pong'

@app.tool(name='Sum_matrix')
def matrix_sum(A:list, B:list):
    '''Aqui se calcula a soma de duas ou mais matrizes. O usuário fornece as matrizes.
    Para calcular, as matrizes precisam estar como *LISTAS*. A função Retornará
    A soma de duas matrizes ou mais. Caso as matrizes possuam tamanhos diferentes, retornará uma mensagem de erro que diz que as matrizes têm tamanhos diferentes'''
    try:
        A=np.array(A)
        B=np.array(B)
        if A.shape != B.shape:
            raise Exception('Não foi possível calcular: Matrizes com tamanhos diferentes')
        sum=A+B
        #print(sum)
        return sum
    except Exception as e:
        return e.args

@app.tool(name='Sub_matrix')
def matrix_sub(A:list, B:list):
    '''Aqui se calcula a subtração de duas ou mais matrizes. O usuário fornece as matrizes.
    Para calcular, as matrizes precisam estar como *LISTAS*. A função Retornará
    A Subtração de duas matrizes ou mais. Caso as matrizes possuam tamanhos diferentes, retornará uma mensagem de erro que diz que as matrizes têm tamanhos diferentes'''
    try:
        A=np.array(A)
        B=np.array(B)
        if A.shape != B.shape:
            raise Exception('Não foi possível calcular: Matrizes com tamanhos diferentes')
        sub=A-B
        return sub
    except Exception as e:
        return e.args

@app.tool(name='mult_by_an_integer')
def matrix_mult_by_integer(A:list, num: float):#errada

    A=np.array(A)
    mult=A*num
    return mult

@app.tool(name='transpose_matrix')
def transpose_matrix(A: list):
    A=np.array(A)
    A=A.transpose()
    return A

@app.tool(name='is_matrix_equals')
def is_matrix_equals(A:list, B:list):
    A=np.array(A)
    B=np.array(B)
    if np.array_equal(A,B):
        return True
    else:
        return False

@app.tool(name='get_matrix_shape')
def get_matrix_shape(A:list):
    A=np.array(A)
    return A.shape

@app.tool(name='mult_matrix')
def mult_matrix(A:list, B:list):
    try:
        A=np.array(A)
        B=np.array(B)
        if A.shape[1] == B.shape[0]:
            mult=np.dot(A,B)
            '''mult=np.zeros(A.shape[0], B.shape[1])
            sec=[]
            for line in range(0,mult.shape[0]+1):
                first=A[line]
                for element in range(0,mult.shape[1]+1):
                    sec.append(B[line][element])
            sum=0
            for i in len(first):
                sum+=(first[i]*sec[i])
            mult'''
            return mult
        else:
            raise Exception('Não foi possível calcular: Tamanhos não batem')
    except Exception as e:
        return e.args
    
@app.tool(name='is_matrix_quad')
def is_matrix_quad(A:list):
    A=np.array(A)
    if A.shape[0]==A.shape[1]:
        return True
    else:
        return False

@app.tool(name='is_matrix_null')
def is_matrix_null(A:list):
    A=np.array(A)
    A=np.sum(A)
    if A==0:
        return True
    else:
        return False
    
@app.tool(name='is_matrix_column')
def is_matrix_column(A:list):
    A=np.array(A)
    if A.shape[1]==1:
        return True
    else:
        return False
    
@app.tool(name='is_matrix_line')
def is_matrix_line(A:list):
    A=np.array(A)
    if A.shape[0]==1:
        return True
    else:
        return False
    
@app.tool(name='is_matrix_diagonal')
def is_matrix_diagonal(A:list):
    if is_matrix_quad(A):#and is_matrix_null(A)
        A=np.array(A)
        for line in range(0,A.shape[0]):
            for column in range(0,A.shape[1]):
                if line!=column and A[line][column]!=0:
                    return False
                
        return True
    else:
        return False
    
@app.tool(name='is_matrix_id_quad')
def is_matrix_id_quad(A:list):
    if is_matrix_quad(A):
        A=np.array(A)
        for line in range(0,A.shape[0]):
            for column in range(0,A.shape[1]):
                if (line!=column and A[line][column]!=0) or (line==column and A[line][column]!=1):
                    return False
        return True
    else:
        return False
    
@app.tool(name='is_matrix_tri_sup')
def is_matrix_tri_sup(A:list):
    if is_matrix_quad(A):
        A=np.array(A)
        for line in range(0,A.shape[0]):
            for column in range (0,A.shape[1]):
                if line>column and A[line][column]!=0:
                    return False
        return True
    else:
        return False
    
@app.tool(name='is_matrix_tri_inf')
def is_matrix_tri_inf(A:list):
    if is_matrix_quad(A):
        A=np.array(A)
        for line in range(0,A.shape[0]):
            for column in range (0,A.shape[1]):
                if line<column and A[line][column]!=0:
                    return False
        return True
    else:
        return False
    
@app.tool(name='is_matrix_simetric')
def is_matrix_simetric(A:list):
    A=np.array(A)
    return np.array_equal(A,transpose_matrix(A))

@app.tool(name='get_matrix_trace')
def get_matrix_trace(A:list):
    try:
        if is_matrix_quad(A):
            A=np.array(A)
            sum=0
            for line in range(0,A.shape[0]):
                for column in range(0,A.shape[1]):
                    if line==column:
                        sum+=A[line][column]
            return sum
        else:
            raise Exception('Não foi possível calcular: A matriz não é quadrada')
    except Exception as e:
        return e.args

@app.tool(name='la_place')
def la_place(A:list):
    try:
        if is_matrix_quad(A):
            A=np.array(A)
            mins=[]
            mins.append(list(np.min(A, axis=0)))
            mins.append(list(np.min(A, axis=1)))
            min=sys.maxsize
            pos=0#posicao da linha/coluna
            axis=0#eixo (0 para linha | 1 para coluna)
            for listmin in range(0,len(mins)):
                for summin in range(0,len(mins[listmin])):
                    if mins[listmin][summin] < min:
                        min=mins[listmin][summin]
                        axis = listmin
                        pos = summin
                    '''if listmin==1:
                        axis=listmin
                    pos=summin'''
            sum=0
            for line in range(0,A.shape[0]):
                for column in range(0,A.shape[1]):
                    if (axis==0 and pos == line) or (axis==1 and pos==column):
                        cofactor = pow(-1, line + column) * minor_entrance(A, line, column)
                        sum += A[line][column] * cofactor
            return sum
            
        else: raise Exception('Não foi possível calcular: A Matriz não é quadrada')
    except Exception as e: return e.args 

@app.tool(name='minor_entrance')
def minor_entrance(A:list, line: int, column: int):
    A=np.array(A)
    #A=np.delete(A, line, axis=0)
    #A=np.delete(A, column, axis=1)
    A_new = A[np.arange(A.shape[0]) != line][:, np.arange(A.shape[1]) != column]

    if A_new.shape==(1,1):
        return A_new[0][0]
    else: return la_place(A_new)

@app.tool(name='get_cofactor')
def get_cofactor(A:list, line:int, column:int):
    try:
        A=np.array(A)
        if line >= A.shape[0] or column >= A.shape[1] or line < 0 or column < 0:
            raise ValueError("Índices fora dos limites da matriz")
        else:
            cofactor = pow(-1, line + column) * minor_entrance(A, line, column)
            return cofactor
    except ValueError as e:
        return e.args
    
@app.tool(name='gauss_elimination_solve')
def gauss_elimination_solve(A: list, b: list):
    """
    Resolve o sistema linear Ax = b usando Eliminação Gaussiana
    com pivoteamento parcial e substituição regressiva.

    Parâmetros:
        A (list): matriz dos coeficientes (n x n) - lista de listas
        b (list): vetor dos termos independentes (n elementos)

    Retorna:
        list: vetor solução x
    """
    A=np.array(A)
    b=np.array(b)
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    # Matriz aumentada
    amp = np.hstack((A, b))

    # Eliminação Gaussiana (triangularização com pivoteamento parcial)
    for k in range(n - 1):
        # Pivoteamento parcial
        max_index = np.argmax(np.abs(amp[k:n, k])) + k
        amp[[k, max_index]] = amp[[max_index, k]]

        for i in range(k + 1, n):
            m = amp[i, k] / amp[k, k]
            amp[i, k:] -= m * amp[k, k:]

    # Substituição regressiva
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = np.dot(amp[i, i+1:n], x[i+1:n])
        x[i] = (amp[i, -1] - soma) / amp[i, i]
    x=x.tolist()
    return x

@app.tool(name='gauss_jordan_solve')
def gauss_jordan_solve(A: list, b: list):#-> np.ndarray
    """
    Resolve um sistema linear Ax = b usando eliminação de Gauss-Jordan com pivoteamento parcial.

    Parâmetros:
        A (list): Matriz dos coeficientes (n x n)
        b (list): Vetor coluna dos resultados (n x 1)

    Retorna:
        list: Vetor solução x
    """
    A=np.array(A)
    b=np.array(b)
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    
    # Matriz aumentada
    amp = np.hstack((A, b))
    
    # Eliminação para forma triangular superior
    for k in range(n - 1):
        # Pivoteamento parcial
        max_index = np.argmax(np.abs(amp[k:n, k])) + k
        amp[[k, max_index]] = amp[[max_index, k]]

        for i in range(k + 1, n):
            m = amp[i, k] / amp[k, k]
            amp[i, :] -= m * amp[k, :]
    
    # Retrosubstituição (escalonamento reverso)
    for k in range(n - 1, -1, -1):
        for i in range(k - 1, -1, -1):
            m = amp[i, k] / amp[k, k]
            amp[i, :] -= m * amp[k, :]
    
    # Normalização das linhas (deixa a diagonal com 1)
    for i in range(n):
        amp[i, :] /= amp[i, i]
    
    # Extração da solução
    x = amp[:, -1]
    x=x.tolist()
    return x

@app.tool(name='cramer_rule')
def cramer_rule(A, b):
    """
    Resolve o sistema Ax = b usando a Regra de Cramer.

    Parâmetros:
    - A: np.ndarray (n x n), matriz dos coeficientes
    - b: np.ndarray (n,), vetor dos termos independentes

    Retorna:
    - x: np.ndarray, vetor solução
    """

    A = A.astype(float)
    b = b.astype(float).flatten()
    n = A.shape[0]

    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0):
        raise ValueError("A matriz A é singular ou quase singular. Regra de Cramer não pode ser aplicada.")

    x = np.zeros(n)
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        det_A_i = np.linalg.det(A_i)
        x[i] = det_A_i / det_A

    return x

@app.tool(name='gauss_seidel_general')
def gauss_seidel_general(A, b, x0=None, tol=1e-5, max_iter=1000):
    """
    Resolve o sistema linear Ax = b pelo método iterativo de Gauss-Seidel.

    Parâmetros:
    - A: np.ndarray (n x n), matriz dos coeficientes
    - b: np.ndarray (n,), vetor dos termos independentes
    - x0: np.ndarray (n,), vetor inicial de aproximação (se None, usa vetor zeros)
    - tol: float, tolerância para critério de parada
    - max_iter: int, número máximo de iterações

    Retorna:
    - x: np.ndarray, vetor solução aproximada
    - n_iter: int, número de iterações realizadas
    """

    A = A.astype(float)
    b = b.astype(float).flatten()
    n = len(b)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.astype(float).flatten()

    for iteration in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            soma1 = np.dot(A[i, :i], x[:i])       # valores atualizados já calculados na iteração atual
            soma2 = np.dot(A[i, i+1:], x_old[i+1:])  # valores ainda da iteração anterior
            x[i] = (b[i] - soma1 - soma2) / A[i, i]

        erro = np.linalg.norm(x - x_old, ord=np.inf)  # erro máximo (norma infinito)

        # Print opcional do progresso
        print(f"Iteração {iteration}: x = {x}, erro = {erro:.6e}")

        if erro < tol:
            print("Convergiu!")
            return x, iteration

    print("Não convergiu no número máximo de iterações.")
    return x, max_iter

@app.tool(name='gauss_jacobi_general')
def gauss_jacobi_general(A, b, x0=None, tol=1e-5, max_iter=1000):
    """
    Resolve o sistema linear Ax = b pelo método iterativo de Gauss-Jacobi.

    Parâmetros:
    - A: np.ndarray (n x n), matriz dos coeficientes
    - b: np.ndarray (n,), vetor dos termos independentes
    - x0: np.ndarray (n,), vetor inicial de aproximação (se None, usa vetor zeros)
    - tol: float, tolerância para critério de parada
    - max_iter: int, número máximo de iterações

    Retorna:
    - x: np.ndarray, vetor solução aproximada
    - n_iter: int, número de iterações realizadas
    """

    A = A.astype(float)
    b = b.astype(float).flatten()
    n = len(b)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.astype(float).flatten()

    for iteration in range(1, max_iter + 1):
        x_old = x.copy()
        x_new = np.zeros_like(x)

        for i in range(n):
            soma = 0
            for j in range(n):
                if j != i:
                    soma += A[i, j] * x_old[j]
            x_new[i] = (b[i] - soma) / A[i, i]

        erro = np.linalg.norm(x_new - x_old, ord=np.inf)

        # Print opcional do progresso
        print(f"Iteração {iteration}: x = {x_new}, erro = {erro:.6e}")

        if erro < tol:
            print("Convergiu!")
            return x_new, iteration

        x = x_new

    print("Não convergiu no número máximo de iterações.")
    return x, max_iter

@app.tool(name='rank_of_reduced_matrix')
def rank_of_reduced_matrix(R, tol=1e-12):
    """
    Calcula o posto de uma matriz já reduzida (forma escalonada)
    contando o número de linhas que não são zero (com tolerância).

    Parâmetros:
    - R: np.ndarray, matriz já reduzida (m x n)
    - tol: float, tolerância para considerar uma linha como nula

    Retorna:
    - rank: int, posto da matriz
    """
    rank = 0
    for row in R:
        if np.any(np.abs(row) > tol):
            rank += 1
    return rank

@app.tool(name='nullity_of_matrix')
def nullity_of_matrix(R, tol=1e-12):
    """
    Calcula a nulidade (dimensão do núcleo) de uma matriz reduzida.

    Parâmetros:
    - R: np.ndarray, matriz reduzida (forma escalonada), tamanho m x n
    - tol: float, tolerância para considerar linha como nula

    Retorna:
    - nulidade: int, dimensão do espaço nulo
    """
    n_cols = R.shape[1]
    rank = rank_of_reduced_matrix(R, tol)
    return n_cols - rank

@app.tool(name='adjoint_matrix')
def adjoint_matrix(A:list):
    try:
        if is_matrix_quad(A):
            A=np.array(A)
            cofactors = np.zeros_like(A)
            for line in (0,A.shape[0]):
                for column in (0,A.shape[1]):
                    cofactors[line][column]=get_cofactor(A, line, column)
            return cofactors.T
        else:
            raise Exception('Não foi possível calcular: matrix não quadrada')
    except Exception as e:
        return e.args

@app.tool(name='inverse_matrix')
def inverse_matrix(A:list):
    try:
        if is_matrix_quad(A):
            A=np.array(A)
            return np.linalg.inv(A)
        else:
            raise ValueError('Não foi possível calcular: matrix não quadrada')
    except ValueError as ve:
        return ve.args
    except np.LinAlgError as lae:
        return lae
    except Exception as e:
        return f'Erro: {e}'

@app.tool(name='swap_rows')    
def swap_rows(matrix, i, j):
    """
    Troca a linha i com a linha j de uma matriz.

    Parâmetros:
    - matrix: np.ndarray (m x n), matriz original
    - i: índice da primeira linha (0-based)
    - j: índice da segunda linha (0-based)

    Retorna:
    - nova matriz com as linhas trocadas
    """
    matrix = np.array(matrix)
    if i == j:
        return matrix  # Nada a fazer
    matrix[[i, j]] = matrix[[j, i]]
    return matrix

@app.tool(name='multiply_row')
def multiply_row(matrix, i, c):
    """
    Multiplica a linha i da matriz pelo escalar c.

    Parâmetros:
    - matrix: np.ndarray (m x n), matriz original
    - i: índice da linha a ser multiplicada (0-based)
    - c: escalar (float) ≠ 0

    Retorna:
    - nova matriz com a linha i multiplicada por c
    """
    try:
        if c == 0:
            raise ValueError("O escalar multiplicador não pode ser zero.")

        matrix = np.array(matrix)
        matrix[i] *= c
        return matrix
    except ValueError as ve:
        return ve.args
    except Exception as e:
        return f'Erro: {e}'

@app.tool(name='add_multiple_of_row')
def add_multiple_of_row(matrix, i, j, c):
    """
    Substitui a linha i por (linha i + c * linha j).

    Parâmetros:
    - matrix: np.ndarray (m x n), matriz original
    - i: índice da linha a ser substituída (0-based)
    - j: índice da linha a ser multiplicada e somada (0-based)
    - c: escalar real (float) ≠ 0

    Retorna:
    - nova matriz com a operação aplicada
    """
    if i == j:
        raise ValueError("Os índices das linhas i e j devem ser diferentes.")
    if c == 0:
        raise ValueError("O escalar c deve ser diferente de zero.")
    
    matrix = np.array(matrix, dtype=float).copy()
    matrix[i] = matrix[i] + c * matrix[j]
    return matrix

@app.tool(name='vector_dimension')
def vector_dimension(vetor: list):
    """
    Retorna a dimensão (número de componentes) de um vetor.

    Parâmetros:
    - vetor: array-like (lista, tupla ou np.ndarray)

    Retorna:
    - int: número de dimensões do vetor (quantidade de componentes)
    """
    vetor = np.array(vetor).flatten()
    return vetor.shape[0]

@app.tool(name='is_vector_equal')
def is_vector_equal(A:list,B:list):
    A=np.array(A).flatten()
    B=np.array(B).flatten()
    if np.equal(A,B):
        return True
    else: return False

@app.tool(name='is_vector_null')
def is_vector_null(A:list):
    A=np.array(A).flatten()
    B=np.zeros_like(A)
    if np.equal(A,B):
        return True
    else: return False

@app.tool(name='vector_sum')
def vector_sum(A:list, B:list):
    '''Aqui se calcula a soma de duas ou mais matrizes. O usuário fornece as matrizes.
    Para calcular, as matrizes precisam estar como *LISTAS*. A função Retornará
    A soma de duas matrizes ou mais. Caso as matrizes possuam tamanhos diferentes, retornará uma mensagem de erro que diz que as matrizes têm tamanhos diferentes'''
    try:
        A=np.array(A).flatten()
        B=np.array(B).flatten()
        if A.shape[0] != B.shape[0]:
            raise Exception('Não foi possível calcular: Matrizes com tamanhos diferentes')
        sum=A+B
        #print(sum)
        return sum
    except Exception as e:
        return e.args

@app.tool(name='vector_sub')
def vector_sub(A:list, B:list):
    '''Aqui se calcula a subtração de duas ou mais matrizes. O usuário fornece as matrizes.
    Para calcular, as matrizes precisam estar como *LISTAS*. A função Retornará
    A Subtração de duas matrizes ou mais. Caso as matrizes possuam tamanhos diferentes, retornará uma mensagem de erro que diz que as matrizes têm tamanhos diferentes'''
    try:
        A=np.array(A).flatten()
        B=np.array(B).flatten()
        if A.shape[0] != B.shape[0]:
            raise Exception('Não foi possível calcular: Matrizes com tamanhos diferentes')
        sub=A-B
        return sub
    except Exception as e:
        return e.args

if __name__=="__main__":
    app.run(transport='stdio')