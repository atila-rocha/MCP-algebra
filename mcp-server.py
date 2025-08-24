from fastmcp import FastMCP
#from mcp.server.fastmcp import FastMCP
import numpy as np
import sys
import math
from math import pow
import random
from scipy.linalg import null_space

app=FastMCP('operadores de matrizes')

@app.tool(name='ping')
def ping():
    '''Caso o usuário escrever ping, retornará pong, isso significará ao usuário que o servidor MCP está funcionando e retornando o que foi pedido'''
    return 'pong'

@app.tool(name='Sum_matrix')
def matrix_sum(A:list, B:list):
    '''
    A função 'matrix_sum' recebe duas matrizes, A e B, no formato de listas aninhadas.
    Ela calcula a soma de A e B e retorna a matriz resultante.

    Parâmetros:
    - A (list): A primeira matriz, fornecida como uma lista aninhada.
    - B (list): A segunda matriz, fornecida como uma lista aninhada.

    Retorna:
    - list: A matriz resultante da soma.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A)
        B=np.array(B)
        if A.shape != B.shape:
            raise ValueError('Erro: As matrizes possuem tamanhos diferentes e não podem ser somadas.')
        sum=A+B
        #print(sum)
        return sum.tolist()
    except ValueError as e:
        return f'Erro: {e}'
    except Exception as e:
        return f'Erro: {e}'

@app.tool(name='Sub_matrix')
def matrix_sub(A:list, B:list):
    '''
    A função 'matrix_sub' recebe duas matrizes, A e B, no formato de listas aninhadas,
    calcula a subtração de A por B e retorna a matriz resultante.

    Parâmetros:
    - A (list): A matriz da qual a subtração será feita, fornecida como uma lista aninhada.
    - B (list): A matriz a ser subtraída, fornecida como uma lista aninhada.

    Retorna:
    - list: A matriz resultante da subtração.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A)
        B=np.array(B)
        if A.shape != B.shape:
            raise Exception('Erro: As matrizes possuem tamanhos diferentes e não podem ser subtraídas.')
        sub=A-B
        return sub.tolist()
    except ValueError as e:
        return f'Erro: {e}'
    except Exception as e:
        return f'Erro: {e}'

@app.tool(name='matrix_mult_by_scalar')
def matrix_mult_by_scalar(A:list, num: float):
    '''
    A função 'matrix_mult_by_scalar' multiplica cada elemento de uma matriz por um número escalar e
    retorna a matriz resultante.

    Parâmetros:
    - A (list): A matriz a ser multiplicada, fornecida como uma lista aninhada de números.
    - num (float): O número escalar pelo qual a matriz será multiplicada.

    Retorna:
    - list: A matriz resultante da multiplicação.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A, dtype=float)
        mult=A*num
        return mult.tolist()
    except (ValueError, TypeError):
        # Captura erros se a entrada A não puder ser convertida para um array de floats
        # ou se 'num' não for um tipo numérico.
        return 'Erro: A matriz ou o escalar não são valores numéricos válidos. Certifique-se de que a matriz contém apenas números e o escalar é um número.'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='transpose_matrix')
def transpose_matrix(A: list):
    '''
    A função 'transpose_matrix' calcula a transposta de uma matriz e a retorna.

    Parâmetros:
    - A (list): A matriz a ser transposta, fornecida como uma lista aninhada.

    Retorna:
    - list: A matriz transposta.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A)
        A=A.transpose()
        return A.tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada A não puder ser convertida para um array de floats,
        # indicando uma entrada inválida.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='is_matrix_equals')
def is_matrix_equals(A:list, B:list):
    '''
    A função 'is_matrix_equals' compara duas matrizes para verificar se elas são idênticas.
    A comparação considera tanto os valores quanto as dimensões das matrizes.

    Parâmetros:
    - A (list): A primeira matriz, fornecida como uma lista aninhada.
    - B (list): A segunda matriz, fornecida como uma lista aninhada.

    Retorna:
    - bool: Retorna True se as matrizes forem idênticas; caso contrário, retorna False.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A)
        B=np.array(B)
        if np.array_equal(A,B):
            return True
        else:
            return False
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats,
        # indicando uma entrada inválida.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='get_matrix_shape')
def get_matrix_shape(A:list):
    '''
    A função 'get_matrix_shape' recebe uma matriz no formato de lista aninhada e retorna suas dimensões.
    As dimensões são retornadas como uma tupla no formato (linhas, colunas).

    Parâmetros:
    - A (list): A matriz para a qual a dimensão será calculada, fornecida como uma lista aninhada.

    Retorna:
    - tuple: Uma tupla contendo o número de linhas e colunas da matriz.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A)
        return A.shape
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada A não puder ser convertida para um array,
        # indicando uma entrada inválida.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='mult_matrix')
def mult_matrix(A:list, B:list):
    '''
    A função 'mult_matrix' realiza a multiplicação da matriz A pela matriz B.
    Para que a multiplicação seja possível, o número de colunas da matriz A
    deve ser igual ao número de linhas da matriz B.

    Parâmetros:
    - A (list): A primeira matriz, fornecida como uma lista aninhada.
    - B (list): A segunda matriz, fornecida como uma lista aninhada.

    Retorna:
    - list: A matriz resultante da multiplicação.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A)
        B=np.array(B)
        if A.shape[1] != B.shape[0]:
            raise ValueError('Não foi possível calcular: Tamanhos não batem')
        mult=np.dot(A,B)
        return mult.tolist()        
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas A ou B não puderem ser convertidas para arrays de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'
    
@app.tool(name='is_matrix_quad')
def is_matrix_quad(A:list):
    '''
    A função 'is_matrix_quad' verifica se uma matriz é quadrada.
    Uma matriz é considerada quadrada se o seu número de linhas for igual ao seu número de colunas.

    Parâmetros:
    - A (list): A matriz a ser verificada, fornecida como uma lista aninhada.

    Retorna:
    - bool: Retorna True se a matriz for quadrada; caso contrário, retorna False.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A)
        if len(A.shape) != 2:
            raise TypeError('Erro: A entrada fornecida não é uma matriz bidimensional. Não é possível verificar se é quadrada.')
        if A.shape[0]==A.shape[1]:
            return True
        else:
            return False
    except (ValueError, TypeError) as e:
    # Captura erros se a entrada A não puder ser convertida para um array de floats,
    # indicando uma entrada inválida.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='is_matrix_null')
def is_matrix_null(A:list):
    '''
    A função 'is_matrix_null' verifica se todos os elementos de uma matriz são zero.

    Parâmetros:
    - A (list): A matriz a ser verificada, fornecida como uma lista aninhada.

    Retorna:
    - bool: Retorna True se a matriz for nula (todos os elementos são zero). Caso contrário, retorna False.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A)
        if np.all(A == 0):
                return True
        else:
            return False
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada A não puder ser convertida para um array de floats,
        # indicando uma entrada inválida.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'
    
@app.tool(name='is_matrix_column')
def is_matrix_column(A:list):
    '''
    A função 'is_matrix_column' verifica se uma matriz é uma matriz coluna.
    Uma matriz é considerada coluna se tiver apenas uma coluna.

    Parâmetros:
    - A (list): A matriz a ser verificada, fornecida como uma lista aninhada.

    Retorna:
    - bool: Retorna True se a matriz for uma matriz coluna; caso contrário, retorna False.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A)
        if len(A.shape) != 2:
            raise ValueError('Erro: A entrada não é uma matriz bidimensional. Não é possível verificar se é uma matriz coluna.')
        if A.shape[1]==1:
            return True
        else:
            return False
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada A não puder ser convertida para um array de floats,
        # indicando uma entrada inválida.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'
    
@app.tool(name='is_matrix_line')
def is_matrix_line(A: list):
    '''
    A função 'is_matrix_line' verifica se uma matriz é uma matriz de linha.
    Uma matriz é considerada de linha se tiver apenas uma linha.

    Parâmetros:
    - A (list): A matriz a ser verificada, fornecida como uma lista aninhada.

    Retorna:
    - bool: Retorna True se a matriz for uma matriz de linha; caso contrário, retorna False.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Tenta converter a lista para um array do NumPy
        np_A = np.array(A, dtype=float)

        # Verifica se a dimensão da matriz é 2D
        if len(np_A.shape) != 2:
            raise ValueError('Erro: A entrada não é uma matriz bidimensional. Não é possível verificar se é uma matriz de linha.')

        # Verifica se o número de linhas é 1
        if np_A.shape[0] == 1:
            return True
        else:
            return False
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada A não puder ser convertida para um array de floats,
        # indicando uma entrada inválida.
        return f'Erro: {e}.'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'
    
@app.tool(name='is_matrix_diagonal')
def is_matrix_diagonal(A: list):
    '''
    A função 'is_matrix_diagonal' verifica se uma matriz é diagonal.
    Uma matriz é considerada diagonal se for quadrada e todos os elementos
    fora da diagonal principal forem zero.

    Parâmetros:
    - A (list): A matriz a ser verificada, fornecida como uma lista aninhada.

    Retorna:
    - bool: Retorna True se a matriz for diagonal; caso contrário, retorna False.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Tenta converter a lista para um array do NumPy
        np_A = np.array(A, dtype=float)

        # Verifica se a matriz é quadrada e tem pelo menos 2 dimensões
        if len(np_A.shape) != 2 or np_A.shape[0] != np_A.shape[1]:
            raise TypeError('A Matriz não é bidimensional ou não é quadrada')

        # Verifica se todos os elementos fora da diagonal principal são zero
        # Para isso, criamos uma máscara booleana e verificamos se todos os elementos
        # correspondentes na matriz original são zero.
        # np.diag(np_A) extrai a diagonal, e np.diag(np.diag(np_A))
        # cria uma nova matriz diagonal a partir dela. A comparação com a
        # matriz original verifica se todos os elementos fora da diagonal são zero.
        return np.all(np_A == np.diag(np.diag(np_A)))
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada A não puder ser convertida para um array de floats,
        # indicando uma entrada inválida.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'
    
@app.tool(name='is_matrix_id_quad')
def is_matrix_id_quad(A: list):
    '''
    A função 'is_matrix_id_quad' verifica se uma matriz é uma matriz identidade.

    Parâmetros:
    - A (list): A matriz a ser verificada, fornecida como uma lista aninhada.

    Retorna:
    - bool: Retorna True se a matriz for uma matriz identidade; caso contrário, retorna False.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Tenta converter a lista para um array do NumPy
        np_A = np.array(A, dtype=float)

        # A matriz identidade deve ser quadrada e ter pelo menos 2 dimensões.
        if len(np_A.shape) != 2 or np_A.shape[0] != np_A.shape[1]:
            raise TypeError('A Matriz não é bidimensional ou não é quadrada')

        # Cria uma matriz identidade do mesmo tamanho que a matriz de entrada
        identity_matrix = np.identity(np_A.shape[0])

        # Compara a matriz de entrada com a matriz identidade
        if np.array_equal(np_A, identity_matrix):
            return True
        else:
            return False

    except (ValueError, TypeError):
        # Captura erros se a entrada A não puder ser convertida para um array de floats,
        # indicando uma entrada inválida.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'
    
@app.tool(name='is_matrix_tri_sup')
def is_matrix_tri_sup(A:list):
    '''
    A função 'is_matrix_tri_sup' verifica se uma matriz é triangular superior.
    Uma matriz é triangular superior se for quadrada e todos os elementos
    abaixo da diagonal principal forem zero.

    Parâmetros:
    - A (list): A matriz a ser verificada, fornecida como uma lista aninhada.

    Retorna:
    - bool: Retorna True se a matriz for triangular superior; caso contrário, retorna False.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Tenta converter a lista para um array do NumPy
        np_A = np.array(A, dtype=float)

        # A matriz triangular superior deve ser quadrada
        if len(np_A.shape) != 2 or np_A.shape[0] != np_A.shape[1]:
            raise TypeError('A Matriz não é bidimensional ou não é quadrada')

        # Verifica se todos os elementos abaixo da diagonal principal são zero
        # np.triu(np_A) cria uma nova matriz com zeros abaixo da diagonal
        # e o np.all() verifica se a matriz original é igual a essa nova matriz.
        return np.all(np_A == np.triu(np_A))

    except (ValueError, TypeError) as e:
        # Captura erros se a entrada A não puder ser convertida para um array de floats,
        # indicando uma entrada inválida.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'
    
@app.tool(name='is_matrix_tri_inf')
def is_matrix_tri_inf(A:list):
    '''
    A função 'is_matrix_tri_inf' verifica se uma matriz é triangular inferior.
    Uma matriz é triangular inferior se for quadrada e todos os elementos
    acima da diagonal principal forem zero.

    Parâmetros:
    - A (list): A matriz a ser verificada, fornecida como uma lista aninhada.

    Retorna:
    - bool: Retorna True se a matriz for triangular inferior; caso contrário, retorna False.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Tenta converter a lista para um array do NumPy
        np_A = np.array(A, dtype=float)

        # A matriz triangular inferior deve ser quadrada
        if len(np_A.shape) != 2 or np_A.shape[0] != np_A.shape[1]:
            raise TypeError('A Matriz não é bidimensional ou não é quadrada')

        # np.tril(np_A) cria uma nova matriz com zeros acima da diagonal.
        # np.all() verifica se a matriz original é igual a essa nova matriz.
        return np.all(np_A == np.tril(np_A))

    except (ValueError, TypeError) as e:
        # Captura erros se a entrada A não puder ser convertida para um array de floats,
        # indicando uma entrada inválida.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'
    
@app.tool(name='is_matrix_simetric')
def is_matrix_simetric(A:list):
    '''
    A função 'is_matrix_simetric' verifica se uma matriz é simétrica.
    Uma matriz é considerada simétrica se for quadrada e igual à sua transposta.

    Parâmetros:
    - A (list): A matriz a ser verificada, fornecida como uma lista aninhada.

    Retorna:
    - bool: Retorna True se a matriz for simétrica; caso contrário, retorna False.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Tenta converter a lista para um array do NumPy
        np_A = np.array(A, dtype=float)

        # A matriz simétrica deve ser quadrada
        if len(np_A.shape) != 2 or np_A.shape[0] != np_A.shape[1]:
            raise TypeError('A Matriz não é bidimensional ou não é quadrada')

        # Verifica se a matriz é igual à sua transposta
        if np.array_equal(np_A, np_A.transpose()):
            return True
        else:
            return False

    except (ValueError, TypeError) as e:
        # Captura erros se a entrada A não puder ser convertida para um array de floats,
        # indicando uma entrada inválida.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='get_matrix_trace')
def get_matrix_trace(A:list):
    '''
    A função 'get_matrix_trace' calcula e retorna o traço de uma matriz.
    O traço de uma matriz é a soma dos elementos em sua diagonal principal.
    A matriz deve ser quadrada para que o traço possa ser calculado.

    Parâmetros:
    - A (list): A matriz para a qual o traço será calculado, fornecida como uma lista aninhada.

    Retorna:
    - float: A soma dos elementos da diagonal principal da matriz.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Tenta converter a lista para um array do NumPy
        np_A = np.array(A, dtype=float)

        # O traço só pode ser calculado para matrizes quadradas
        if len(np_A.shape) != 2 or np_A.shape[0] != np_A.shape[1]:
            raise TypeError('Erro: Não foi possível calcular. O traço só é definido para matrizes quadradas.')

        # Usa a função np.trace() para calcular o traço de forma eficiente
        trace_value = np.trace(np_A)
        return float(trace_value)
    
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada A não puder ser convertida para um array de floats,
        # indicando uma entrada inválida.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='la_place')
def la_place(A:list):
    '''
    A função 'la_place' calcula o determinante de uma matriz usando o Teorema de Laplace.
    O método é recursivo e deve ser aplicado apenas a matrizes quadradas.

    Parâmetros:
    - A (list): A matriz para a qual o determinante será calculado, fornecida como uma lista aninhada.

    Retorna:
    - float: O valor do determinante da matriz.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
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
            
        else: raise TypeError('Não foi possível calcular: A Matriz não é quadrada')
    except (TypeError, ValueError) as e: return f'Erro: {e}'
    except Exception as e: return f'Erro inesperado: {e}'

@app.tool(name='minor_entrance')
def minor_entrance(A:list, line: int, column: int):
    '''
    A função 'minor_entrance' calcula o menor complementar (ou menor de entrada) de uma matriz
    com relação a uma linha e coluna específicas. Isso é útil, por exemplo, no cálculo do
    determinante por cofactores (regra de Laplace).

    Parâmetros:
    - A (list): Matriz original representada como uma lista de listas.
    - line (int): Índice da linha a ser removida.
    - column (int): Índice da coluna a ser removida.

    Retorna:
    - float ou int: O menor complementar correspondente à posição (line, column). Se a matriz
      resultante tiver tamanho 1x1, retorna o valor diretamente. Caso contrário, a função 
      chama 'la_place' para calcular o determinante da submatriz resultante.

    Observações:
    - A função pressupõe que a função auxiliar 'la_place' já está definida e implementa 
      o cálculo do determinante de forma recursiva (regra de Laplace).
    - A entrada A deve representar uma matriz numérica válida e de dimensão compatível
      com os índices informados.
    '''

    A=np.array(A)
    #A=np.delete(A, line, axis=0)
    #A=np.delete(A, column, axis=1)
    A_new = A[np.arange(A.shape[0]) != line][:, np.arange(A.shape[1]) != column]

    if A_new.shape==(1,1):
        return A_new[0][0]
    else: return la_place(A_new)

@app.tool(name='get_cofactor')
def get_cofactor(A:list, line:int, column:int):
    '''
    A função 'get_cofactor' calcula o cofator de um elemento específico de uma matriz.
    O cofator é usado principalmente no cálculo de determinantes e na inversa de matrizes,
    seguindo a Regra de Laplace.

    Parâmetros:
    - A (list): A matriz original representada como uma lista de listas (matriz aninhada).
    - line (int): Índice da linha do elemento cujo cofator se deseja calcular.
    - column (int): Índice da coluna do elemento.

    Retorna:
    - float ou int: O valor do cofator do elemento na posição (line, column), calculado como:
      \[
      \text{cofator} = (-1)^{\text{line} + \text{column}} \times M
      \]
      onde \( M \) é o menor complementar, calculado pela função `minor_entrance`.

    - str: Uma mensagem de erro explicando o que deu errado, caso a entrada seja inválida
      ou ocorra algum erro inesperado.

    Validações:
    - Verifica se os índices fornecidos estão dentro dos limites da matriz.
    - Lança um erro informativo se os índices forem inválidos ou se a matriz não for bem formada.

    Observações:
    - A função depende da função auxiliar `minor_entrance`, que retorna o menor complementar
      da matriz.
    - Essa função é útil no contexto de álgebra linear, especialmente para expandir determinantes
      ou calcular matrizes adjuntas.
    '''

    try:
        A=np.array(A)
        if line >= A.shape[0] or column >= A.shape[1] or line < 0 or column < 0:
            raise ValueError("Índices fora dos limites da matriz")
        else:
            cofactor = pow(-1, line + column) * minor_entrance(A, line, column)
            return cofactor
    except ValueError as e:
        return f'Erro: {e}'
    except Exception as e:
        return f'Erro inesperado: {e}'
    
@app.tool(name='gauss_elimination_solve')
def gauss_elimination_solve(A: list, b: list):
    '''
    Resolve o sistema linear Ax = b usando Eliminação Gaussiana
    com pivoteamento parcial e substituição regressiva.

    Parâmetros:
    - A (list): Matriz dos coeficientes (n x n) - lista de listas.
    - b (list): Vetor dos termos independentes (n elementos).

    Retorna:
    - list: O vetor solução x.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A)
        b=np.array(b)
        A = A.astype(float)
        b = b.astype(float)
        if len(A.shape) != 2 or A.shape[0] != A.shape[1] or A.shape[0] != b.size:
            raise TypeError('Erro: As dimensões da matriz A e do vetor b não são compatíveis. A deve ser uma matriz quadrada e o número de linhas deve ser igual ao número de elementos em b.')
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
    except (TypeError, ValueError) as e:
        return f'Erro: {e}'
    except Exception as e:
        return f'Erro inesperado: {e}'

@app.tool(name='gauss_jordan_solve')
def gauss_jordan_solve(A: list, b: list):#-> np.ndarray
    '''
    Resolve um sistema linear Ax = b usando Eliminação de Gauss-Jordan com
    pivoteamento parcial. A matriz de coeficientes deve ser quadrada.

    Parâmetros:
    - A (list): Matriz dos coeficientes (n x n) - lista de listas.
    - b (list): Vetor dos termos independentes (n elementos).

    Retorna:
    - list: O vetor solução x.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A)
        b=np.array(b)
        A = A.astype(float)
        b = b.astype(float)
        if len(A.shape) != 2 or A.shape[0] != A.shape[1] or A.shape[0] != b.size:
            raise TypeError('Erro: As dimensões da matriz A e do vetor b não são compatíveis. A deve ser uma matriz quadrada e o número de linhas deve ser igual ao número de elementos em b.')
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
    except (TypeError, ValueError) as e:
        return f'Erro: {e}'
    except Exception as e:
        return f'Erro inesperado: {e}'

@app.tool(name='cramer_rule')
def cramer_rule(A, b):
    '''
    Resolve o sistema linear Ax = b usando a Regra de Cramer.
    Este método é eficiente para matrizes de pequeno porte.

    Parâmetros:
    - A (list): Matriz dos coeficientes (n x n) - lista de listas.
    - b (list): Vetor dos termos independentes (n elementos).

    Retorna:
    - list: O vetor solução x.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A = A.astype(float)
        b = b.astype(float).flatten()
        if len(A.shape) != 2 or A.shape[0] != A.shape[1] or A.shape[0] != b.size:
            raise TypeError('Erro: As dimensões da matriz A e do vetor b não são compatíveis. A deve ser uma matriz quadrada e o número de linhas deve ser igual ao número de elementos em b.')

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

        return x.tolist()
    except (TypeError, ValueError) as e:
        return f'Erro: {e}'
    except Exception as e:
        return f'Erro inesperado: {e}'

@app.tool(name='gauss_seidel_general')
def gauss_seidel_general(A, b, x0=None, tol=1e-5, max_iter=1000):
    '''
    Resolve o sistema linear Ax = b usando o método iterativo de Gauss-Seidel.
    O método é mais eficiente para matrizes estritamente dominantes pela diagonal.

    Parâmetros:
    - A (list): Matriz dos coeficientes (n x n) - lista de listas.
    - b (list): Vetor dos termos independentes (n elementos).
    - x0 (list): Vetor inicial de aproximação (se None, usa vetor zeros).
    - tol (float): Tolerância para o critério de parada.
    - max_iter (int): Número máximo de iterações.

    Retorna:
    - tuple: Um tupla contendo o vetor solução aproximada (list) e o número de iterações realizadas (int).
    - str: Uma mensagem de erro clara se a operação falhar ou não convergir.
    '''
    try:
        A = A.astype(float)
        b = b.astype(float).flatten()
        if len(A.shape) != 2 or A.shape[0] != A.shape[1] or A.shape[0] != b.size:
            raise TypeError('Erro: As dimensões da matriz A e do vetor b não são compatíveis. A deve ser uma matriz quadrada e o número de linhas deve ser igual ao número de elementos em b.')
        n = len(b)

        diag = np.diag(np.abs(A))
        off_diag_sum = np.sum(np.abs(A), axis=1) - diag
        if np.any(diag <= off_diag_sum):
            raise ValueError("Erro: A matriz não é estritamente dominante pela diagonal, a convergência não é garantida. O método pode falhar.")

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

        raise ValueError(f'Não convergiu no número máximo de iterações. Iteração: {max_iter}, x: {x}')
    except (TypeError, ValueError) as e:
        return f'Erro: {e}'
    except Exception as e:
        return f'Erro inesperado: {e}'

@app.tool(name='gauss_jacobi_general')
def gauss_jacobi_general(A, b, x0=None, tol=1e-5, max_iter=1000):
    '''
    Resolve o sistema linear Ax = b usando o método iterativo de Gauss-Jacobi.
    O método é mais eficiente para matrizes estritamente dominantes pela diagonal.

    Parâmetros:
    - A (list): Matriz dos coeficientes (n x n) - lista de listas.
    - b (list): Vetor dos termos independentes (n elementos).
    - x0 (list): Vetor inicial de aproximação (se None, usa vetor zeros).
    - tol (float): Tolerância para o critério de parada.
    - max_iter (int): Número máximo de iterações.

    Retorna:
    - tuple: Um tupla contendo o vetor solução aproximada (list) e o número de iterações realizadas (int).
    - str: Uma mensagem de erro clara se a operação falhar ou não convergir.
    '''
    try:
        A = A.astype(float)
        b = b.astype(float).flatten()
        if len(A.shape) != 2 or A.shape[0] != A.shape[1] or A.shape[0] != b.size:
            raise TypeError('Erro: As dimensões da matriz A e do vetor b não são compatíveis. A deve ser uma matriz quadrada e o número de linhas deve ser igual ao número de elementos em b.')
        n = len(b)

        diag = np.diag(np.abs(A))
        off_diag_sum = np.sum(np.abs(A), axis=1) - diag
        if np.any(diag <= off_diag_sum):
            raise ValueError("Erro: A matriz não é estritamente dominante pela diagonal, a convergência não é garantida. O método pode falhar.")

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

        raise ValueError(f'Não convergiu no número máximo de iterações. Iteração: {max_iter}, x: {x}')
    except (TypeError, ValueError) as e:
        return f'Erro: {e}'
    except Exception as e:
        return f'Erro inesperado: {e}'

@app.tool(name='rank_of_reduced_matrix')
def rank_of_reduced_matrix(R, tol=1e-12):
    '''
    Calcula o posto de uma matriz já reduzida (forma escalonada)
    contando o número de linhas que não são zero (com tolerância).

    Parâmetros:
    - R (list): A matriz escalonada, fornecida como uma lista aninhada.
    - tol (float): A tolerância para considerar uma linha como nula.

    Retorna:
    - int: O posto da matriz.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        np_R = np.array(R, dtype=float)
        rank = 0
        if len(np_R.shape) != 2:
                raise TypeError("Erro: A entrada não é uma matriz bidimensional válida.")
        for row in np_R:
            if np.any(np.abs(row) > tol):
                rank += 1
        return rank
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='nullity_of_matrix')
def nullity_of_matrix(R, tol=1e-12):
    '''
    Calcula a nulidade (dimensão do espaço nulo) de uma matriz reduzida.
    A nulidade é calculada usando o Teorema do Posto-Nulidade:
    nulidade = número de colunas - posto.

    Parâmetros:
    - R (list): A matriz escalonada, fornecida como uma lista aninhada.
    - tol (float): A tolerância para considerar uma linha como nula no cálculo do posto.

    Retorna:
    - int: A dimensão do espaço nulo da matriz.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        np_R = np.array(R, dtype=float)

            # Verifica se a entrada é uma matriz bidimensional
        if len(np_R.shape) != 2:
            raise TypeError("Erro: A entrada não é uma matriz bidimensional válida.")
        n_cols = np_R.shape[1]
        rank = rank_of_reduced_matrix(np_R, tol)
        return int(n_cols - rank)
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='adjoint_matrix')
def adjoint_matrix(A:list):
    '''
    A função 'adjoint_matrix' calcula a matriz adjunta de uma matriz quadrada.
    A matriz adjunta é a transposta da matriz de cofatores.
    Este método é eficiente para matrizes de pequeno porte.

    Parâmetros:
    - A (list): A matriz quadrada, fornecida como uma lista aninhada.

    Retorna:
    - list: A matriz adjunta.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        if is_matrix_quad(A):
            A=np.array(A)
            cofactors = np.zeros_like(A)
            for line in (0,A.shape[0]):
                for column in (0,A.shape[1]):
                    cofactors[line][column]=get_cofactor(A, line, column)
            return cofactors.T.tolist()
        else:
            raise TypeError('Não foi possível calcular: matrix não quadrada')
    except (TypeError, ValueError) as e:
        return f'Erro: {e}'
    except Exception as e:
        return f'Erro inesperado: {e}'

@app.tool(name='inverse_matrix')
def inverse_matrix(A:list):
    try:
        if is_matrix_quad(A):
            A=np.array(A)
            return np.linalg.inv(A).tolist()
        else:
            raise ValueError('Não foi possível calcular: matrix não quadrada')
    except ValueError as ve:
        return f'Erro: {ve}'
    except np.LinAlgError as lae:
        return f'Erro no cálculo: {lae}'
    except Exception as e:
        return f'Erro inesperado: {e}'

@app.tool(name='swap_rows')    
def swap_rows(matrix, i, j):
    '''
    Troca a linha i com a linha j de uma matriz.

    Parâmetros:
    - matrix (list): A matriz a ser modificada, fornecida como uma lista aninhada.
    - i (int): O índice da primeira linha (base 0).
    - j (int): O índice da segunda linha (base 0).

    Retorna:
    - list: A nova matriz com as linhas trocadas.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        matrix = np.array(matrix, dtype=float)
        if len(matrix.shape) != 2:
            raise TypeError("A entrada não é uma matriz bidimensional válida.")
        num_rows = matrix.shape[0]
            # Verifica se os índices estão dentro dos limites da matriz
        if not (0 <= i < num_rows and 0 <= j < num_rows):
            raise TypeError(f"Os índices de linha devem estar entre 0 e {num_rows - 1}.")
        if i == j:
            return matrix.tolist()  # Nada a fazer
        matrix[[i, j]] = matrix[[j, i]]
        return matrix.tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats,
        # indicando uma entrada inválida.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='multiply_row')
def multiply_row(matrix, i, c):
    '''
    Multiplica a linha i da matriz pelo escalar c.
    Esta é uma das operações elementares de linha usadas para resolver sistemas lineares.

    Parâmetros:
    - matrix (list): A matriz a ser modificada, fornecida como uma lista aninhada.
    - i (int): O índice da linha a ser multiplicada (base 0).
    - c (float): O escalar multiplicador.

    Retorna:
    - list: A nova matriz com a linha i multiplicada por c.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        matrix = np.array(matrix)
        if len(matrix.shape) != 2:
            raise TypeError("A entrada não é uma matriz bidimensional válida.")
        num_rows = matrix.shape[0]
            # Verifica se os índices estão dentro dos limites da matriz
        if not (0 <= i < num_rows):
            raise TypeError(f"Os índices de linha devem estar entre 0 e {num_rows - 1}.")
        if c == 0:
            raise ValueError("O escalar multiplicador não pode ser zero.")
        matrix[i] *= c
        return matrix.tolist()
    except ValueError as ve:
        return f'Erro: {ve}'
    except Exception as e:
        return f'Erro inesperado: {e}'

@app.tool(name='add_multiple_of_row')
def add_multiple_of_row(matrix, i, j, c):
    '''
    Substitui a linha i por (linha i + c * linha j).
    Esta é uma das operações elementares de linha usadas para resolver sistemas lineares
    e encontrar a forma escalonada de uma matriz.

    Parâmetros:
    - matrix (list): A matriz a ser modificada, fornecida como uma lista aninhada.
    - i (int): O índice da linha a ser substituída (base 0).
    - j (int): O índice da linha a ser multiplicada e somada (base 0).
    - c (float): O escalar real multiplicador.

    Retorna:
    - list: A nova matriz com a operação aplicada.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        matrix = np.array(matrix, dtype=float).copy()
        if len(matrix.shape) != 2:
                raise TypeError("A entrada não é uma matriz bidimensional válida.")
        num_rows = matrix.shape[0]
            # Verifica se os índices estão dentro dos limites da matriz
        if not (0 <= i < num_rows and 0 <= j < num_rows):
            raise TypeError(f"Os índices de linha devem estar entre 0 e {num_rows - 1}.")
        if i == j:
            raise ValueError("Os índices das linhas i e j devem ser diferentes.")
        if c == 0:
            raise ValueError("O escalar c deve ser diferente de zero.")
        matrix[i] = matrix[i] + c * matrix[j]
        return matrix.tolist()
    except ValueError as ve:
        return f'Erro: {ve}'
    except Exception as e:
        return f'Erro inesperado: {e}'

@app.tool(name='vector_dimension')
def vector_dimension(vetor: list):
    '''
    A função 'vector_dimension' retorna a dimensão (número de componentes) de um vetor.

    Parâmetros:
    - vector (list): O vetor a ser analisado, fornecido como uma lista.

    Retorna:
    - int: O número de componentes do vetor.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        vetor = np.array(vetor, dtype=float).flatten()
        return vetor.shape[0]
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='is_vector_equal')
def is_vector_equal(A:list,B:list):
    '''
    A função 'is_vector_equal' verifica se dois vetores são iguais.
    A comparação é feita elemento a elemento.

    Parâmetros:
    - A (list): O primeiro vetor.
    - B (list): O segundo vetor.

    Retorna:
    - bool: Retorna True se os vetores forem iguais; caso contrário, retorna False.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A, dtype=float).flatten()
        B=np.array(B, dtype=float).flatten()
        if A.shape != B.shape:
            raise TypeError('Os vetores têm dimensões diferentes.')
        if np.all(A == B):
            return True, 'Os vetores são iguais.'
        else:
            return False, 'Os vetores são diferentes.'
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para arrays de floats.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='is_vector_null')
def is_vector_null(A:list):
    '''
    A função 'is_vector_null' verifica se um vetor é o vetor nulo (ou seja, se todos os seus componentes são zero).

    Parâmetros:
    - A (list): O vetor a ser verificado, fornecido como uma lista.

    Retorna:
    - bool: Retorna True se o vetor for nulo; caso contrário, retorna False.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A).flatten()
        if A.size == 0:
            raise TypeError('Erro: O vetor não pode ser vazio.')
        if np.all(np.isclose(A, 0)):
            return True
        else: return False
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='vector_sum')
def vector_sum(A:list, B:list):
    '''
    A função 'vector_sum' calcula a soma de dois vetores.

    Para que a soma seja possível, os vetores devem ter a mesma dimensão.
    Esta função manipula as entradas como vetores (listas de 1 dimensão),
    retornando um novo vetor que é a soma dos componentes correspondentes.

    Parâmetros:
    - A (list): O primeiro vetor, fornecido como uma lista.
    - B (list): O segundo vetor, fornecido como uma lista.

    Retorna:
    - list: O vetor resultante da soma.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A, dtype=float).flatten()
        B=np.array(B, dtype=float).flatten()
        if A.shape[0] != B.shape[0]:
            raise TypeError('Não foi possível calcular: Matrizes com tamanhos diferentes')
        sum=A+B
        #print(sum)
        return sum.tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='vector_sub')
def vector_sub(A:list, B:list):
    '''
    A função 'vector_sub' calcula a subtração de dois vetores.

    Para que a subtração seja possível, os vetores devem ter a mesma dimensão.
    Esta função manipula as entradas como vetores (listas de 1 dimensão),
    retornando um novo vetor que é a subtração dos componentes correspondentes.

    Parâmetros:
    - A (list): O primeiro vetor, fornecido como uma lista.
    - B (list): O segundo vetor, fornecido como uma lista.

    Retorna:
    - list: O vetor resultante da subtração.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        A=np.array(A,dtype=float).flatten()
        B=np.array(B,dtype=float).flatten()
        if A.shape[0] != B.shape[0]:
            raise TypeError('Não foi possível calcular: Matrizes com tamanhos diferentes')
        sub=A-B
        return sub.tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='vector_mult_by_an_integer')
def vector_mult_by_an_integer(vector, scalar):
    '''
    A função 'vector_mult_by_an_integer' multiplica um vetor por um escalar.
    A multiplicação é feita elemento a elemento.

    Parâmetros:
    - vector (list): O vetor a ser multiplicado, fornecido como uma lista.
    - scalar (float): O escalar (número real) pelo qual o vetor será multiplicado.

    Retorna:
    - list: O vetor resultante da multiplicação.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        vector = np.array(vector, dtype=float).flatten()
        scalar = float(scalar)  # Garante que seja um número real
        if vector.size == 0:
            raise ValueError('Erro: O vetor não pode ser vazio.')
        result = vector * scalar
        return result.tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para tipos numéricos.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='is_vector_parallel')
def is_vector_parallel(v1, v2, tol=1e-9):
    '''
    Verifica se dois vetores são paralelos (colineares).
    Dois vetores são paralelos se um for um múltiplo escalar do outro.

    Parâmetros:
    - v1 (list): O primeiro vetor.
    - v2 (list): O segundo vetor.
    - tol (float): Tolerância para comparação de ponto flutuante.

    Retorna:
    - tuple: (bool, str) - True e uma mensagem se forem paralelos;
             False e uma mensagem se não forem.
    '''
    try:
        # 1. Converte as listas para vetores NumPy 1D (achatados)
        v1 = np.array(v1).flatten()
        v2 = np.array(v2).flatten()

        # 2. Verifica se os dois vetores têm a mesma dimensão
        if v1.shape != v2.shape:
            raise TypeError(' Os vetores têm dimensões diferentes e, portanto, não podem ser paralelos.')  # Vetores com dimensões diferentes não podem ser colineares

        if np.allclose(v1, 0, atol=tol):
            if np.allclose(v2, 0, atol=tol):
                return 'Ambos os vetores são o vetor nulo, portanto, são paralelos.'
            else:
                return 'O primeiro vetor é nulo, mas o segundo não. Eles não são paralelos.'

        # 4. Tenta encontrar a razão entre cada par de componentes correspondentes (v2[i] / v1[i])
        razoes = []
        for a, b in zip(v1, v2):
            if abs(a) < tol:
                if abs(b) > tol:
                    return False  # Um é zero, o outro não → não são colineares
                else:
                    continue  # Ambos zero → ainda possível serem colineares
            razoes.append(b / a)

        is_parallel = np.allclose(razoes, razoes[0], atol=tol)
        if is_parallel:
            return True
        else:
            return False
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return False, f'Erro inesperado: {e}'

@app.tool(name='get_vector_components_from_not_original_point')
def get_vector_components_from_not_original_point(ponto_P, ponto_Q):
    '''
    A função 'get_vector_components_from_not_original_point' calcula as componentes de um vetor
    cujo ponto inicial não é a origem.

    O vetor é definido pela diferença das coordenadas do ponto final (Q) e do ponto inicial (P),
    ou seja, o vetor PQ = Q - P.

    Parâmetros:
    - ponto_P (list): As coordenadas do ponto inicial P, fornecidas como uma lista.
    - ponto_Q (list): As coordenadas do ponto final Q, fornecidas como uma lista.

    Retorna:
    - list: O vetor resultante das componentes (Q - P).
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        P = np.array(ponto_P, dtype=float).flatten()
        Q = np.array(ponto_Q, dtype=float).flatten()


        if P.shape != Q.shape:
            raise ValueError("Os pontos devem ter a mesma dimensão.")
        vector = Q - P
        return vector.tolist()
    except (ValueError, TypeError)as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='vector_norm')
def vector_norm(vetor):
    '''
    A função 'vector_norm' calcula a norma Euclidiana (módulo ou magnitude) de um vetor.
    A norma de um vetor é o seu comprimento, calculado como a raiz quadrada da soma
    dos quadrados de seus componentes.

    Parâmetros:
    - vector (list): O vetor a ser analisado, fornecido como uma lista.

    Retorna:
    - float: A norma do vetor.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        vetor = np.array(vetor, dtype=float).flatten()
        norm = np.linalg.norm(vetor)
        return float(norm)
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='unit_vector')
def unit_vector(vetor):
    '''
    A função 'unit_vector' normaliza um vetor, retornando o seu vetor unitário.
    Um vetor unitário tem a mesma direção e sentido do vetor original,
    mas com magnitude igual a 1.

    Parâmetros:
    - vector (list): O vetor a ser normalizado, fornecido como uma lista.

    Retorna:
    - list: O vetor unitário.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        vetor = np.array(vetor, dtype=float).flatten()
        norma = np.linalg.norm(vetor)

        if np.isclose(norma, 0):
            raise TypeError("Erro: O vetor nulo não pode ser normalizado.")

        unit_vector = vetor / norma
        return unit_vector.tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats.
        return f'Erro: {e}.'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='is_unit_vector')
def is_unit_vector(vetor, tol=1e-8):
    '''
    A função 'is_unit_vector' verifica se um vetor é um vetor unitário.
    Um vetor é unitário se a sua norma (magnitude) for igual a 1, com uma
    pequena tolerância para evitar problemas de ponto flutuante.

    Parâmetros:
    - vector (list): O vetor a ser verificado, fornecido como uma lista.
    - tol (float): A tolerância para a comparação com 1.

    Retorna:
    - tuple: (bool, str) - True e uma mensagem se for um vetor unitário;
             False e uma mensagem caso contrário.
    '''
    try:
        vetor = np.array(vetor, dtype=float).flatten()
        if vetor.size == 0:
            raise TypeError('O vetor não pode ser vazio.')
        norma = np.linalg.norm(vetor)

        is_unit = np.isclose(norma, 1, atol=tol)

        if is_unit:
            return True
        else:
            return False
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='dif_between_two_points')
def dif_between_two_points(p1:list, p2:list):
    '''
    A função 'dif_between_two_points' calcula a distância Euclidiana entre dois pontos.

    A distância é calculada como a norma do vetor obtido pela subtração das
    coordenadas dos pontos, ou seja, a norma de (p2 - p1).

    Parâmetros:
    - p1 (list): As coordenadas do primeiro ponto, fornecidas como uma lista.
    - p2 (list): As coordenadas do segundo ponto, fornecidas como uma lista.

    Retorna:
    - float: A distância entre os dois pontos.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        p1 = np.array(p1, dtype=float).flatten()
        p2 = np.array(p2, dtype=float).flatten()

        if p1.shape != p2.shape:
            raise ValueError("Os pontos devem ter a mesma dimensão para calcular a distância.")

        # Calcula a diferença entre os pontos e, em seguida, a norma do vetor resultante
        distance = np.linalg.norm(p2 - p1)
        return float(distance)
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='vectors_scale_procuct')
def vectors_scale_procuct(v1, v2):
    '''
    A função 'vectors_scale_procuct' calcula o produto escalar (dot product) de dois vetores.
    O produto escalar só pode ser calculado entre vetores que têm a mesma dimensão.

    Parâmetros:
    - v1 (list): O primeiro vetor, fornecido como uma lista.
    - v2 (list): O segundo vetor, fornecido como uma lista.

    Retorna:
    - float: O valor do produto escalar.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        v1 = np.array(v1, dtype=float).flatten()
        v2 = np.array(v2, dtype=float).flatten()

        if v1.shape != v2.shape:
            raise ValueError("Vetores devem ter a mesma dimensão para o produto escalar.")

        # Calcula o produto escalar
        dot_product = np.dot(v1, v2)

        return float(dot_product)
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='produto_elemento_a_elemento')
def produto_elemento_a_elemento(u, v):
    """
    Retorna o produto elemento a elemento (Hadamard) entre dois vetores.

    Args:
        u (array-like): Primeiro vetor.
        v (array-like): Segundo vetor.

    Returns:
        np.ndarray: Um vetor com o produto u[i] * v[i] para cada i.

    Raises:
        ValueError: Se os vetores tiverem tamanhos diferentes.
    """
    try:
        u = np.array(u, dtype=float)
        v = np.array(v, dtype=float)
    
        if u.shape != v.shape:
            raise ValueError("Os vetores devem ter o mesmo tamanho.")
    
        return u * v
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='get_angle_between_two_vectors')
def get_angle_between_two_vectors(u, v, em_graus=False):
    '''
    A função 'get_angle_between_two_vectors' calcula o ângulo entre dois vetores.
    O ângulo é calculado usando a fórmula do produto escalar: cos(theta) = (u . v) / (||u|| * ||v||).

    Parâmetros:
    - u (list): O primeiro vetor.
    - v (list): O segundo vetor.
    - em_graus (bool): Se for True, retorna o ângulo em graus. Caso contrário, retorna em radianos.

    Retorna:
    - float: O ângulo entre os vetores (em radianos ou graus, dependendo do parâmetro).
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        u = np.array(u, dtype=float).flatten()
        v = np.array(v, dtype=float).flatten()

        if u.shape != v.shape:
            raise ValueError("Os vetores devem ter a mesma dimensão para calcular o ângulo entre eles.")

        norma_u = np.linalg.norm(u)
        norma_v = np.linalg.norm(v)

        if np.isclose(norma_u, 0) or np.isclose(norma_v, 0):
            raise ValueError("Não é possível calcular o ângulo com vetor nulo.")

        produto = np.dot(u, v)
        cos_theta = produto / (norma_u * norma_v)

        # Correção numérica para evitar erro de domínio
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angulo_rad = np.arccos(cos_theta)

        if em_graus:
            return float(np.degrees(angulo_rad))
        return float(angulo_rad)
    except (ValueError, TypeError) as e:
    # Captura erros se as entradas não puderem ser convertidas para vetores numéricos.
        return f'Erro: {e}'
    except Exception as e:
    # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='vectors_scale_procuct_by_angle')
def vectors_scale_procuct_by_angle(u, v, angle, in_degrees=False):
    '''
    A função 'vectors_scalar_product_by_angle' calcula o produto escalar de dois vetores
    usando a fórmula: u . v = ||u|| * ||v|| * cos(theta), onde theta é o ângulo entre eles.

    Parâmetros:
    - u (list): O primeiro vetor.
    - v (list): O segundo vetor.
    - angle (float): O ângulo entre os vetores.
    - in_degrees (bool): Se for True, assume que o ângulo está em graus. Caso contrário, assume radianos.

    Retorna:
    - float: O valor do produto escalar.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        u = np.array(u, dtype=float).flatten()
        v = np.array(v, dtype=float).flatten()
        angle=float(angle)

        if u.shape != v.shape:
            raise ValueError("Os vetores devem ter a mesma dimensão para o produto escalar.")

        norma_u = np.linalg.norm(u)
        norma_v = np.linalg.norm(v)

        if np.isclose(norma_u, 0) or np.isclose(norma_v, 0):
            raise ValueError("Não é possível calcular o produto escalar com um vetor nulo.")

        # Converte o ângulo para radianos, se necessário
        if in_degrees:
            angle = np.radians(angle)

        result = norma_u * norma_v * np.cos(angle)

        return float(result)
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para tipos numéricos
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='is_vector_orthogonal')
def is_vector_orthogonal(u, v, tol=1e-10):
    '''
    A função 'is_vector_orthogonal' verifica se dois vetores são ortogonais.
    Dois vetores são ortogonais se o seu produto escalar for zero.
    
    Parâmetros:
    - u (list): O primeiro vetor.
    - v (list): O segundo vetor.
    - tol (float): Uma pequena tolerância para erros de ponto flutuante.

    Retorna:
    - tuple: (bool, str) - True e uma mensagem se forem ortogonais;
             False e uma mensagem caso contrário.
    '''
    try:
        u = np.array(u, dtype=float).flatten()
        v = np.array(v, dtype=float).flatten()

        if u.shape != v.shape:
            raise ValueError("Os vetores devem ter a mesma dimensão para a verificação de ortogonalidade.")

        dot_product = np.dot(u, v)

        is_orthogonal = np.isclose(dot_product, 0, atol=tol)

        if is_orthogonal:
            return True
        else:
            return False
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='eq_point_normal')
def eq_point_normal(ponto:list, vetor_normal:list):
    '''
    A função 'eq_point_normal' gera os coeficientes da equação ponto-normal para
    qualquer dimensão. A equação ponto-normal é dada por n . (X - P0) = 0,
    que pode ser expandida para ax + by + cz + ... + d = 0, onde n = [a, b, c, ...].
    
    Parâmetros:
    - point (list): As coordenadas do ponto P0.
    - normal_vector (list): As coordenadas do vetor normal n.

    Retorna:
    - dict: Um dicionário contendo o vetor normal, o ponto e o termo d da equação.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        ponto = np.array(ponto, dtype=float).flatten()
        vetor_normal = np.array(vetor_normal, dtype=float).flatten()

        if ponto.shape != vetor_normal.shape:
            raise ValueError("O ponto e o vetor normal devem ter o mesmo número de dimensões.")

        # Calcula o termo d
        d = -np.dot(vetor_normal, ponto)

        # Cria dicionário com coeficientes
        coeficientes = {
            'normais': vetor_normal.tolist(),
            'ponto': ponto.tolist(),
            'd': float(d)
        }
        return coeficientes
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos.
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='vector_horizontal_projection')
def vector_horizontal_projection(u: list, a: list):
    '''
    A função 'vector_horizontal_projection' calcula a projeção horizontal do vetor u no vetor a.
    A projeção de u em a é um vetor que tem a mesma direção do vetor a e
    magnitude igual à sombra de u no vetor a.
    
    A fórmula da projeção é: proj_a(u) = [(u . a) / ||a||^2] * a
    
    Parâmetros:
    - u (list): O vetor a ser projetado.
    - a (list): O vetor sobre o qual a projeção é feita.

    Retorna:
    - list: O vetor resultante da projeção.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        u = np.array(u, dtype=float).flatten()
        a = np.array(a, dtype=float).flatten()
        if u.shape != a.shape:
            raise TypeError("Os vetores devem ter a mesma dimensão para calcular a projeção.")

        norma_a = np.linalg.norm(a)
        if np.isclose(norma_a, 0):
            raise ValueError("O vetor sobre o qual a projeção é feita não pode ser o vetor nulo.")

        # Calcula o produto escalar entre u e a
        dot_product = np.dot(u, a)

        # Calcula a projeção usando a fórmula
        projection = (dot_product / (norma_a ** 2)) * a

        return projection.tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='vector_vertical_projection')
def vector_vertical_projection(u: list, a: list):
    '''
    A função 'vector_vertical_projection' calcula a projeção vertical do vetor u no vetor a.
    A projeção vertical é a componente de u que é ortogonal ao vetor a.
    
    A fórmula da projeção vertical é: proj_vertical(u) = u - proj_horizontal(u)
    
    Parâmetros:
    - u (list): O vetor a ser projetado.
    - a (list): O vetor sobre o qual a projeção é feita.

    Retorna:
    - list: O vetor resultante da projeção vertical.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        u = np.array(u, dtype=float).flatten()
        a = np.array(a, dtype=float).flatten()
        if u.shape != a.shape:
            raise TypeError("Os vetores devem ter a mesma dimensão para calcular a projeção.")
        projh=vector_vertical_projection(u,a)

        vertical_projection = u - projh

        return vertical_projection.tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='norm_projection_without_angle')
def norm_projection_without_angle(u: list, a: list):
    '''
    A função 'norm_projection_without_angle' calcula a norma (magnitude) da projeção
    do vetor u sobre o vetor a, sem usar o ângulo entre eles.
    
    A fórmula da norma da projeção é: ||proj_a(u)|| = |u . a| / ||a||.
    
    Parâmetros:
    - u (list): O vetor a ser projetado.
    - a (list): O vetor sobre o qual a projeção é feita.

    Retorna:
    - float: A norma da projeção de u sobre a.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converter para numpy arrays
        u = np.array(u, dtype=float).flatten()
        a = np.array(a, dtype=float).flatten()

        # Verificações
        if u.shape != a.shape:
            raise TypeError("Os vetores devem ter a mesma dimensão para calcular a projeção.")

        norma_a = np.linalg.norm(a)
        if np.isclose(norma_a, 0):
            raise TypeError("O vetor sobre o qual a projeção é feita não pode ser o vetor nulo.")

        # Calcular norma da projeção: |u·a| / ||a||
        dot_product = np.dot(u, a)
        norma_proj = abs(dot_product) / norma_a

        return float(norma_proj)
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='norm_projection_with_angle')
def norm_projection_with_angle(u: list, a: list, 
                             angle: float, em_graus: bool = True):
    '''
    A função 'norm_projection_with_angle' calcula a norma (magnitude) da projeção
    do vetor u sobre o vetor a, usando o ângulo entre eles.
    
    A fórmula da norma da projeção é: ||proj_a(u)|| = ||u|| * |cos(theta)|.
    
    Parâmetros:
    - u (list): O vetor a ser projetado.
    - a (list): O vetor sobre o qual a projeção é feita (usado apenas para validação de dimensão).
    - angle (float): O ângulo entre os vetores u e a.
    - in_degrees (bool): Se for True, assume que o ângulo está em graus. Caso contrário, assume radianos.

    Retorna:
    - float: A norma da projeção.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converter para numpy arrays
        u = np.array(u, dtype=float).flatten()
        a = np.array(a, dtype=float).flatten()
        angle=float(angle)

        # Verificações básicas
        if u.shape != a.shape:
            raise ValueError('Os vetores devem ter a mesma dimensão para calcular a projeção.')

        if np.isclose(np.linalg.norm(a), 0):
            raise ValueError("O vetor 'a' não pode ser o vetor nulo")

        # Converter ângulo para radianos se necessário
        if em_graus:
            angulo_rad = np.radians(angle)
        else:
            angulo_rad = angle

        # Calcular norma da projeção: ||u|| * |cos(θ)|
        norma_u = np.linalg.norm(u)
        norma_proj = norma_u * abs(np.cos(angulo_rad))

        return float(norma_proj)
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores ou números
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='generic_distance_point_plan_or_straight')
def generic_distance_point_plan_or_straight(po: list, n: list):
    '''
    A função 'generic_distance_point_plan_or_straight' calcula a distância de um ponto
    a uma reta (2D) ou plano (3D/nD). Também pode ser usada para calcular a distância
    entre dois planos paralelos.
    
    A fórmula utilizada é:
    distância = |a₁x₁ + a₂x₂ + ... + aₙxₙ + d| / √(a₁² + a₂² + ... + aₙ²)
    
    Parâmetros:
    - po (list): Coordenadas do ponto.
    - n (list): Coeficientes da equação da reta/plano, incluindo o termo independente 'd'.

    Retorna:
    - float: A distância calculada.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converter para numpy arrays
        po = np.array(po, dtype=float).flatten()
        n = np.array(n, dtype=float).flatten()

        # Verificar dimensões
        if len(n) != len(po) + 1:
            return TypeError(f"Erro: Para um ponto com {len(po)} dimensões, os coeficientes devem ter {len(po) + 1} elementos. Recebido: {len(n)}.")

            # Separa os coeficientes direcionais e o termo independente
        directional_coeffs = n[:-1]
        independent_term = n[-1]

        # Separar coeficientes direcionais e termo independente


        # Verificar se os coeficientes direcionais não são todos zero
        norma_coef = np.linalg.norm(directional_coeffs)
        if np.isclose(norma_coef, 0):
            raise ValueError("Todos os coeficientes direcionais não podem ser zero")

        # Calcular distância: |ax + by + cz + ... + d| / √(a² + b² + c² + ...)
        numerador = abs(np.dot(directional_coeffs, po) + independent_term)
        denominador = norma_coef

        distance = numerador / denominador

        return float(distance)
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores ou números
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='is_plans_paralels')
def is_plans_paralels(a: list,b: list):
    '''
    A função 'is_plans_paralels' verifica se dois planos são paralelos.
    Dois planos são paralelos se seus vetores normais forem paralelos.
    
    A equação de um plano é: a₁x₁ + a₂x₂ + ... + aₙxₙ + d = 0,
    onde o vetor normal é [a₁, a₂, ..., aₙ].
    
    Parâmetros:
    - a (list): Lista de coeficientes do primeiro plano, incluindo o termo independente.
    - b (list): Lista de coeficientes do segundo plano, incluindo o termo independente.

    Retorna:
    - bool: Retorna True se os planos forem paralelos; caso contrário, retorna False.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        a=np.array(a, dtype=float).flatten()
        b=np.array(b, dtype=float).flatten()
        if len(a) != len(b):
            raise TypeError("Os planos têm dimensões diferentes.")
        normal_a = a[:-1]
        normal_b = b[:-1]
        return is_vector_parallel(normal_a,normal_b)
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores ou números
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='generic_distance_parallel_hyperplanes')
def generic_distance_parallel_hyperplanes(n1: list, n2: list):
    """
    Calcula a distância entre dois hiperplanos paralelos em um espaço n-dimensional.

    A distância entre dois hiperplanos paralelos é definida como a menor distância
    entre qualquer ponto de um deles ao outro. Para isso, a função:
    1. Verifica se os vetores normais dos hiperplanos são paralelos.
    2. Encontra um ponto arbitrário sobre o primeiro hiperplano.
    3. Usa a fórmula da distância ponto-a-plano para calcular a distância desse ponto
       até o segundo hiperplano.

    Fórmula aplicada:
        distância = |n₂ ⋅ p + d₂| / ||n₂||
    onde:
        - n₂ é o vetor normal do segundo hiperplano
        - d₂ é o termo independente do segundo hiperplano
        - p é um ponto qualquer sobre o primeiro hiperplano

    Parâmetros
    ----------
    n1 : list
        Coeficientes do primeiro hiperplano no formato [a1, a2, ..., an, d1],
        onde [a1, ..., an] é o vetor normal e d1 o termo independente.
    
    n2 : list
        Coeficientes do segundo hiperplano no mesmo formato de n1.

    Retorna
    -------
    float
        Distância mínima entre os dois hiperplanos paralelos.

    Levanta
    -------
    ValueError
        - Se os hiperplanos não forem paralelos.
        - Se os vetores normais forem nulos.
        - Se as listas tiverem dimensões diferentes.

    RuntimeError
        - Se ocorrer falha ao encontrar um ponto sobre o primeiro hiperplano
          (situação teórica improvável se os dados forem válidos).
    """
    try:
        n1 = np.array(n1, dtype=float)
        n2 = np.array(n2, dtype=float)

        if len(n1) != len(n2):
            raise ValueError("Os hiperplanos devem ter a mesma dimensão.")

        # Separar o vetor normal e o termo independente
        normal_vector1 = n1[:-1]
        d1 = n1[-1]

        normal_vector2 = n2[:-1]
        d2 = n2[-1]

        # Verifica se os vetores normais não são nulos
        norm1 = np.linalg.norm(normal_vector1)
        norm2 = np.linalg.norm(normal_vector2)

        if np.isclose(norm1, 0) or np.isclose(norm2, 0):
            raise ValueError("O vetor normal não pode ser nulo.")

        # Verifica se os hiperplanos são paralelos
        # Multiplica-se por norma para lidar com vetores múltiplos de forma robusta
        if not is_plans_paralels(normal_vector1, normal_vector2):
            raise ValueError("Os hiperplanos não são paralelos.")

        # 1. Encontrar um ponto P0 no primeiro hiperplano (n1)
        # A dimensão do espaço é o número de coeficientes direcionais.
        dim = len(normal_vector1)
        p0 = np.zeros(dim)

        # Encontra o primeiro coeficiente não nulo e usa-o para encontrar o ponto
        encontrado = False
        for i in range(dim):
            if not np.isclose(normal_vector1[i], 0):
                p0[i] = -d1 / normal_vector1[i]
                encontrado = True
                break

        # Garantia de que um ponto foi encontrado (embora o 'if np.isclose(norm1, 0)'
        # já garanta que pelo menos um coeficiente não é zero)
        if not encontrado:
            raise RuntimeError("Erro interno: não foi possível encontrar um ponto no hiperplano.")

        # 2. Aplicar a fórmula da distância de P0 ao segundo hiperplano (n2)
        # D = |a_2*x_0 + b_2*y_0 + ... + d_2| / ||normal_vector2||

        numerador = abs(np.dot(normal_vector2, p0) + d2)
        denominador = np.linalg.norm(normal_vector2)

        distance = numerador / denominador

        return distance
    except RuntimeError as e:
        return e
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores ou números
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='produto_vetorial')
def produto_vetorial(u:list, v:list):
    '''
    A função 'produto_vetorial' calcula o produto vetorial entre dois vetores 3D.
    
    O produto vetorial retorna um novo vetor que é perpendicular a ambos os vetores
    originais. A operação é definida apenas para vetores em três dimensões.

    Parâmetros:
    - u (list): O primeiro vetor.
    - v (list): O segundo vetor.

    Retorna:
    - list: O vetor resultante do produto vetorial.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        u = np.array(u, dtype=float).flatten()
        v = np.array(v, dtype=float).flatten()

        if u.shape != (3,) or v.shape != (3,):
            raise ValueError(" O produto vetorial só é definido para vetores tridimensionais.")

        cross_product = np.cross(u, v)

        return cross_product.tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='norma_produto_vetorial')
def norma_produto_vetorial(u:list, v:list):
    '''
    A função 'norma_produto_vetorial' calcula a norma (magnitude) do produto vetorial
    entre dois vetores 3D. A norma do produto vetorial é numericamente igual à área
    do paralelogramo formado pelos dois vetores.

    Parâmetros:
    - u (list): O primeiro vetor 3D.
    - v (list): O segundo vetor 3D.

    Retorna:
    - float: A norma do produto vetorial.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        u = np.array(u, dtype=float).flatten()
        v = np.array(v, dtype=float).flatten()
        if u.shape != (3,) or v.shape != (3,):
            raise TypeError("A norma do produto vetorial só é definida para vetores tridimensionais.")
        # Produto vetorial
        cross_product = np.cross(u, v)

        # Norma do vetor resultante
        norm_cross_product = np.linalg.norm(cross_product)

        return float(norm_cross_product)
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='norma_produto_vetorial_angulo')
def norma_produto_vetorial_angulo(u:list, v:list, angulo=None, em_graus=True):
    '''
    A função 'norma_produto_vetorial_angulo' calcula a norma (magnitude) do produto
    vetorial entre dois vetores 3D. A norma do produto vetorial é numericamente igual à
    área do paralelogramo formado pelos dois vetores.
    
    A função pode calcular a norma de duas formas:
    1. A partir dos próprios vetores (se o 'angulo' não for fornecido).
    2. Usando o ângulo fornecido entre os vetores.
    
    Fórmula: ||u × v|| = ||u|| * ||v|| * sin(theta)

    Parâmetros:
    - u (list): O primeiro vetor 3D.
    - v (list): O segundo vetor 3D.
    - angulo (float, opcional): O ângulo entre os vetores. Se 'None', o ângulo será
      calculado a partir dos vetores.
    - em_graus (bool): Se 'True', o ângulo fornecido está em graus. Caso contrário,
      está em radianos. O padrão é 'True'.

    Retorna:
    - float: A norma do produto vetorial.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        u = np.array(u, dtype=float).flatten()
        v = np.array(v, dtype=float).flatten()
        if u.shape != (3,) or v.shape != (3,):
            raise TypeError("A norma do produto vetorial só é definida para vetores tridimensionais.")
        norma_u = np.linalg.norm(u)
        norma_v = np.linalg.norm(v)

        if angulo is None:
            if np.isclose(norma_u, 0) or np.isclose(norma_v, 0):
                return 0.0
            dot = np.dot(u, v)
            cos_theta = dot / (norma_u * norma_v)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            sin_theta = np.sqrt(1 - cos_theta**2)
        else:
            if em_graus:
                angulo = math.radians(angulo)
            sin_theta = math.sin(angulo)
        norma_produto = norma_u * norma_v * sin_theta

        return float(norma_produto)
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='produto_misto')
def produto_misto(u: list, v: list, w: list):
    '''
    A função 'produto_misto' calcula o produto misto de três vetores 3D.
    A fórmula utilizada é: u · (v × w).
    
    O valor absoluto do produto misto representa o volume do paralelepípedo
    formado pelos três vetores.

    Parâmetros:
    - u (list): O primeiro vetor 3D.
    - v (list): O segundo vetor 3D.
    - w (list): O terceiro vetor 3D.

    Retorna:
    - float: O valor do produto misto.
      - Se = 0: os vetores são coplanares (o volume é zero).
      - Se > 0: o sistema é destro (segue a regra da mão direita).
      - Se < 0: o sistema é canhoto.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converter para arrays numpy
        u = np.array(u, dtype=float).flatten()
        v = np.array(v, dtype=float).flatten()
        w = np.array(w, dtype=float).flatten()

        # Verificar se todos são vetores 3D
        if u.shape != (3,) or v.shape != (3,) or w.shape != (3,):
            raise ValueError("Todos os vetores devem ter exatamente 3 dimensões")

        # Calcular produto vetorial v × w
        produto_vetorial = np.cross(v, w)

        # Calcular produto escalar u · (v × w)
        produto_misto_resultado = np.dot(u, produto_vetorial)

        return float(produto_misto_resultado)
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para vetores numéricos
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='criar_matriz_de_transformacao')
def criar_matriz_de_transformacao(matriz_coeficientes: list):
    """
    Gera a matriz de transformação linear a partir de uma matriz de coeficientes quadrada.

    A função interpreta a matriz de coeficientes como a representação matricial
    de uma transformação linear na base canônica, aplicando a transformação a cada
    vetor da base canônica para reconstruir a matriz da transformação.

    Parâmetros:
    -----------
    matriz_coeficientes : list
        Lista (ou array) representando uma matriz quadrada de coeficientes (nxn).
        Cada linha representa um vetor de coeficientes da transformação.

    Retorna:
    --------
    list
        Matriz da transformação linear como uma lista de listas (array 2D).

    Exceções:
    ---------
    Retorna uma mensagem de erro caso:
      - A matriz de coeficientes não seja quadrada.
      - A entrada não possa ser convertida para uma matriz NumPy de floats.
      - Qualquer outro erro inesperado ocorra.

    Notas:
    ------
    - A matriz da transformação gerada corresponde à matriz de coeficientes original,
      pois a transformação é aplicada à base canônica.
    - A função converte a saída para lista para facilitar o uso fora do contexto NumPy.
    """
    try:
        matriz_coeficientes= np.array(matriz_coeficientes, dtype=float)
        if len(matriz_coeficientes.shape) != 2 or matriz_coeficientes.shape[0] != matriz_coeficientes.shape[1]:
            raise TypeError("A matriz de transformação deve ser quadrada para aplicar a transformação na base canônica.")

        # A dimensão é a quantidade de linhas (ou colunas, já que é quadrada).
        dimensao = matriz_coeficientes.shape[0]

        # Cria a matriz identidade, que representa os vetores da base canônica.
        matriz_identidade = np.identity(dimensao)

        # Função interna para aplicar a transformação, usando a matriz de coeficientes.
        def transformacao_interna(vetor):
            # A transformação é a multiplicação da matriz de coeficientes pelo vetor.
            return matriz_coeficientes @ vetor

        # Lista para armazenar as colunas da matriz de transformação.
        colunas_matriz = []

        # Aplica a transformação em cada vetor da base canônica.
        # O loop percorre cada linha da matriz identidade, que é um vetor da base.
        for vetor_base in matriz_identidade:
            vetor_transformado = transformacao_interna(vetor_base)
            colunas_matriz.append(vetor_transformado)

        # Empilha os vetores transformados como colunas para formar a matriz.
        matriz_transformacao = np.column_stack(colunas_matriz)

        return matriz_transformacao.tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='aplicar_transformacao')
def aplicar_transformacao(matriz_transformacao:list, vetor_original:list):
    '''
    A função 'aplicar_transformacao' aplica uma transformação linear a um vetor.
    A transformação é realizada pela multiplicação da matriz de transformação pelo vetor de entrada.
    
    Parâmetros:
    - matriz_transformacao (list): A matriz que define a transformação,
      fornecida como uma lista aninhada. Deve ser uma matriz 2D.
    - vetor_original (list): O vetor a ser transformado,
      fornecido como uma lista de números. Deve ser um array 1D.

    Retorna:
    - list: O vetor resultante da transformação.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        np_matriz = np.array(matriz_transformacao, dtype=float)
        np_vetor = np.array(vetor_original, dtype=float).flatten()
        if len(np_matriz.shape) != 2 or np_matriz.shape[1] != np_vetor.shape[0]:
            raise TypeError("As dimensões da matriz e do vetor não são compatíveis para a multiplicação.")
        # A multiplicação de matrizes em NumPy pode ser feita com o operador @
        # ou com a função np.dot(). O operador @ é mais legível.
        vetor_transformado = np.dot(np_matriz, np_vetor)

        return vetor_transformado.tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se as entradas não puderem ser convertidas para arrays
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='reflexao_eixo_y')
def reflexao_eixo_y(vetor:list):
    """
    Realiza a reflexão de um vetor em relação ao eixo y no espaço 2D.

    A matriz de transformação para a reflexão no eixo y mantém o componente y
    inalterado, enquanto inverte o sinal do componente x.
    
    Args:
        vetor (list): O vetor 2D a ser transformado, por exemplo, np.array([x, y]).
    
    Returns:
        list: Vetor refletido em formato de lista
    """
    try:
        np_vetor = np.array(vetor, dtype=float).flatten()
        if np_vetor.shape[0] != 2:
            raise TypeError("A reflexão no eixo y só é definida para vetores 2D.")
        # Matriz de reflexão em relação ao eixo y
        matriz_reflexao = np.array([
            [-1, 0],
            [0, 1]
        ], dtype=float)

        vetor_transformado = np.dot(matriz_reflexao, np_vetor)

        return vetor_transformado.tolist()
    except (ValueError, TypeError):
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return 'Erro: A entrada fornecida não é um vetor válido. Certifique-se de que é uma lista de números.'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='reflexao_eixo_x')
def reflexao_eixo_x(vetor:list):
    '''
    A função 'reflexao_eixo_x' realiza a reflexão de um vetor em relação ao eixo x no espaço 2D.

    A matriz de transformação para a reflexão no eixo x mantém o componente x
    inalterado, enquanto inverte o sinal do componente y.
    
    Parâmetros:
    - vetor (list): O vetor 2D a ser transformado, por exemplo, [x, y].
    
    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A reflexão no eixo x é definida apenas para vetores 2D
        if np_vetor.shape[0] != 2:
            raise TypeError("A reflexão no eixo x só é definida para vetores 2D.")

        # Matriz de reflexão em relação ao eixo x
        matriz_reflexao = np.array([
            [1, 0],
            [0, -1]
        ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_reflexao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='reflexao_reta_y_x')
def reflexao_reta_y_x(vetor: list):
    '''
    A função 'reflexao_reta_y_x' realiza a reflexão de um vetor em relação à reta y=x no espaço 2D.

    A matriz de transformação para a reflexão na reta y=x troca os componentes
    x e y do vetor.
    
    Parâmetros:
    - vetor (list): O vetor 2D a ser transformado, por exemplo, [x, y].
    
    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A reflexão na reta y=x é definida apenas para vetores 2D
        if np_vetor.shape[0] != 2:
            raise TypeError("A reflexão na reta y=x só é definida para vetores 2D.")

        # Matriz de reflexão em relação à reta y=x
        matriz_reflexao = np.array([
            [0, 1],
            [1, 0]
        ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_reflexao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='reflexao_plano_xy')
def reflexao_plano_xy(vetor: list):
    '''
    A função 'reflexao_plano_xy' realiza a reflexão de um vetor em relação ao plano xy no espaço 3D.

    A matriz de transformação para a reflexão no plano xy mantém os componentes x
    e y inalterados, enquanto inverte o sinal do componente z.
    
    Parâmetros:
    - vetor (list): O vetor 3D a ser transformado, por exemplo, [x, y, z].
    
    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A reflexão no plano xy é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A reflexão no plano xy só é definida para vetores 3D.")

        # Matriz de reflexão em relação ao plano xy
        matriz_reflexao = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_reflexao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='reflexao_plano_xz')
def reflexao_plano_xz(vetor: list):
    '''
    A função 'reflexao_plano_xz' realiza a reflexão de um vetor em relação ao plano xz no espaço 3D.

    A matriz de transformação para a reflexão no plano xz mantém os componentes x
    e z inalterados, enquanto inverte o sinal do componente y.
    
    Parâmetros:
    - vetor (list): O vetor 3D a ser transformado, por exemplo, [x, y, z].
    
    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A reflexão no plano xz é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A reflexão no plano xz só é definida para vetores 3D.")

        # Matriz de reflexão em relação ao plano xz
        matriz_reflexao = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_reflexao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='reflexao_plano_yz')
def reflexao_plano_yz(vetor: list):
    '''
    A função 'reflexao_plano_yz' realiza a reflexão de um vetor em relação ao plano yz no espaço 3D.

    A matriz de transformação para a reflexão no plano yz mantém os componentes y
    e z inalterados, enquanto inverte o sinal do componente x.
    
    Parâmetros:
    - vetor (list): O vetor 3D a ser transformado, por exemplo, [x, y, z].
    
    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A reflexão no plano yz é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A reflexão no plano yz só é definida para vetores 3D.")

        # Matriz de reflexão em relação ao plano yz
        matriz_reflexao = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_reflexao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='projecao_ortogonal_eixo_x')
def projecao_ortogonal_eixo_x(vetor: list):
    '''
    A função 'projecao_ortogonal_eixo_x' realiza a projeção ortogonal de um vetor
    em relação ao eixo x no espaço 2D.

    A matriz de projeção mantém o componente x e zera o componente y.
    
    Parâmetros:
    - vetor (list): O vetor 2D a ser transformado, por exemplo, [x, y].
    
    Retorna:
    - list: O vetor projetado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A projeção no eixo x é definida apenas para vetores 2D
        if np_vetor.shape[0] != 2:
            raise TypeError("A projeção no eixo x só é definida para vetores 2D.")

        # Matriz de projeção ortogonal em relação ao eixo x
        matriz_projecao = np.array([
            [1, 0],
            [0, 0]
        ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_projecao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='projecao_ortogonal_eixo_y')
def projecao_ortogonal_eixo_y(vetor: list):
    '''
    A função 'projecao_ortogonal_eixo_y' realiza a projeção ortogonal de um vetor
    em relação ao eixo y no espaço 2D.

    A matriz de projeção mantém o componente y e zera o componente x.
    
    Parâmetros:
    - vetor (list): O vetor 2D a ser transformado, por exemplo, [x, y].
    
    Retorna:
    - list: O vetor projetado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A projeção no eixo y é definida apenas para vetores 2D
        if np_vetor.shape[0] != 2:
            raise TypeError("A projeção no eixo y só é definida para vetores 2D.")

        # Matriz de projeção ortogonal em relação ao eixo y
        matriz_projecao = np.array([
            [0, 0],
            [0, 1]
        ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_projecao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='reflexao_plano_xy_3d')
def reflexao_plano_xy_3d(vetor: list):
    '''
    A função 'reflexao_plano_xy_3d' realiza a reflexão de um vetor em relação ao plano xy no espaço 3D.

    A matriz de transformação para a reflexão no plano xy mantém os componentes x
    e y inalterados, enquanto inverte o sinal do componente z.
    
    Parâmetros:
    - vetor (list): O vetor 3D a ser transformado, por exemplo, [x, y, z].
    
    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A reflexão no plano xy é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A reflexão no plano xy só é definida para vetores 3D.")

        # Matriz de reflexão em relação ao plano xy
        matriz_reflexao = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_reflexao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='reflexao_plano_xz_3d')
def reflexao_plano_xz_3d(vetor: list):
    '''
    A função 'reflexao_plano_xz_3d' realiza a reflexão de um vetor em relação ao plano xz no espaço 3D.

    A matriz de transformação para a reflexão no plano xz mantém os componentes x
    e z inalterados, enquanto inverte o sinal do componente y.
    
    Parâmetros:
    - vetor (list): O vetor 3D a ser transformado, por exemplo, [x, y, z].
    
    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A reflexão no plano xz é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A reflexão no plano xz só é definida para vetores 3D.")

        # Matriz de reflexão em relação ao plano xz
        matriz_reflexao = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_reflexao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='reflexao_plano_yz_3d')
def reflexao_plano_yz_3d(vetor: list):
    '''
    A função 'reflexao_plano_yz_3d' realiza a reflexão de um vetor em relação ao plano yz no espaço 3D.

    A matriz de transformação para a reflexão no plano yz mantém os componentes y
    e z inalterados, enquanto inverte o sinal do componente x.
    
    Parâmetros:
    - vetor (list): O vetor 3D a ser transformado, por exemplo, [x, y, z].
    
    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A reflexão no plano yz é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A reflexão no plano yz só é definida para vetores 3D.")

        # Matriz de reflexão em relação ao plano yz
        matriz_reflexao = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_reflexao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='projecao_ortogonal_plano_xy')
def projecao_ortogonal_plano_xy(vetor: list):
    '''
    A função 'projecao_ortogonal_plano_xy' realiza a projeção ortogonal de um vetor
    em relação ao plano xy no espaço 3D.

    A matriz de projeção para o plano xy mantém os componentes x e y inalterados,
    enquanto anula o componente z.
    
    Parâmetros:
    - vetor (list): O vetor 3D a ser transformado, por exemplo, [x, y, z].
    
    Retorna:
    - list: O vetor projetado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A projeção é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A projeção no plano xy só é definida para vetores 3D.")

        # Matriz de projeção em relação ao plano xy
        matriz_projecao = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_projecao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='projecao_ortogonal_plano_xz')
def projecao_ortogonal_plano_xz(vetor: list):
    '''
    A função 'projecao_ortogonal_plano_xz' realiza a projeção ortogonal de um vetor
    em relação ao plano xz no espaço 3D.

    A matriz de projeção para o plano xz mantém os componentes x e z inalterados,
    enquanto anula o componente y.
    
    Parâmetros:
    - vetor (list): O vetor 3D a ser transformado, por exemplo, [x, y, z].
    
    Retorna:
    - list: O vetor projetado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A projeção é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A projeção no plano xz só é definida para vetores 3D.")

        # Matriz de projeção em relação ao plano xz
        matriz_projecao = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_projecao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='projecao_ortogonal_plano_yz')
def projecao_ortogonal_plano_yz(vetor: list):
    '''
    A função 'projecao_ortogonal_plano_yz' realiza a projeção ortogonal de um vetor
    em relação ao plano yz no espaço 3D.

    A matriz de projeção para o plano yz mantém os componentes y e z inalterados,
    enquanto anula o componente x.
    
    Parâmetros:
    - vetor (list): O vetor 3D a ser transformado, por exemplo, [x, y, z].
    
    Retorna:
    - list: O vetor projetado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A projeção é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A projeção no plano yz só é definida para vetores 3D.")

        # Matriz de projeção em relação ao plano yz
        matriz_projecao = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_projecao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='rotacao_2d')
def rotacao_2d(vetor: list, angle: float, is_horario=False, is_degrees=True):
    '''
    A função 'rotacao_anti_horario' realiza a rotação de um vetor em torno da origem no plano 2D.
    
    A rotação é anti-horária por padrão, mas pode ser invertida para horária com o parâmetro 'is_horario'.
    O ângulo de entrada pode ser em graus (padrão) ou radianos.
    
    Parâmetros:
    - vetor (list): O vetor 2D a ser transformado, por exemplo, [x, y].
    - angle (float): O valor do ângulo de rotação.
    - is_horario (bool): Se True, a rotação é no sentido horário. O padrão é False.
    - is_degrees (bool): Se True, o ângulo está em graus. O padrão é True.
    
    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()
        
        # A rotação é definida apenas para vetores 2D
        if np_vetor.shape[0] != 2:
            raise TypeError("A rotação é definida apenas para vetores 2D.")

        # Converte o ângulo para radianos se a entrada for em graus
        if is_degrees:
            theta = np.radians(angle)
        else:
            theta = angle
            
        # Matriz de rotação: a lógica da matriz muda dependendo do sentido
        if is_horario:
            # Matriz para rotação no sentido horário
            matriz_rotacao = np.array([
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)]
            ], dtype=float)
        else:
            # Matriz para rotação no sentido anti-horário (padrão)
            matriz_rotacao = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_rotacao, np_vetor)
        
        return vetor_transformado.tolist()
        
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='rotacao_eixo_x')
def rotacao_eixo_x(vetor: list, angle: float, is_horario=False, is_degrees=True):
    '''
    A função 'rotacao_eixo_x' realiza a rotação de um vetor 3D em torno do eixo X positivo.

    Parâmetros:
    - vetor (list): O vetor 3D a ser rotacionado, por exemplo, [x, y, z].
    - angle (float): O valor do ângulo de rotação.
    - is_horario (bool): Se True, a rotação é no sentido horário. O padrão é False.
    - is_degrees (bool): Se True, o ângulo está em graus. O padrão é True.

    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A rotação em 3D é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A rotação em 3D é definida apenas para vetores 3D.")

        # Converte o ângulo para radianos se a entrada for em graus
        if is_degrees:
            theta = np.radians(angle)
        else:
            theta = angle

        # Matriz de rotação em torno do eixo X
        if not is_horario:
            # Matriz de rotação anti-horária
            matriz_rotacao = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ], dtype=float)
        else:
            # Matriz de rotação horária
            matriz_rotacao = np.array([
                [1, 0, 0],
                [0, np.cos(theta), np.sin(theta)],
                [0, -np.sin(theta), np.cos(theta)]
            ], dtype=float)
        
        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_rotacao, np_vetor)

        return vetor_transformado.tolist()

    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='rotacao_eixo_y')
def rotacao_eixo_y(vetor: list, angle: float, is_horario=False, is_degrees=True):
    '''
    A função 'rotacao_eixo_y' realiza a rotação de um vetor 3D em torno do eixo Y positivo.

    Parâmetros:
    - vetor (list): O vetor 3D a ser rotacionado, por exemplo, [x, y, z].
    - angle (float): O valor do ângulo de rotação.
    - is_horario (bool): Se True, a rotação é no sentido horário. O padrão é False.
    - is_degrees (bool): Se True, o ângulo está em graus. O padrão é True.

    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A rotação em 3D é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A rotação em 3D é definida apenas para vetores 3D.")

        # Converte o ângulo para radianos se a entrada for em graus
        if is_degrees:
            theta = np.radians(angle)
        else:
            theta = angle

        # Matriz de rotação em torno do eixo Y
        if not is_horario:
            # Matriz de rotação anti-horária
            matriz_rotacao = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ], dtype=float)
        else:
            # Matriz de rotação horária
            matriz_rotacao = np.array([
                [np.cos(theta), 0, -np.sin(theta)],
                [0, 1, 0],
                [np.sin(theta), 0, np.cos(theta)]
            ], dtype=float)

        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_rotacao, np_vetor)

        return vetor_transformado.tolist()

    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='rotacao_eixo_z')
def rotacao_eixo_z(vetor: list, angle: float, is_horario=False, is_degrees=True):
    '''
    A função 'rotacao_eixo_z' realiza a rotação de um vetor 3D em torno do eixo Z positivo.

    Parâmetros:
    - vetor (list): O vetor 3D a ser rotacionado, por exemplo, [x, y, z].
    - angle (float): O valor do ângulo de rotação.
    - is_horario (bool): Se True, a rotação é no sentido horário. O padrão é False.
    - is_degrees (bool): Se True, o ângulo está em graus. O padrão é True.

    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A rotação em 3D é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A rotação em 3D é definida apenas para vetores 3D.")

        # Converte o ângulo para radianos se a entrada for em graus
        if is_degrees:
            theta = np.radians(angle)
        else:
            theta = angle

        # Matriz de rotação em torno do eixo Z
        if not is_horario:
            # Matriz de rotação anti-horária
            matriz_rotacao = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=float)
        else:
            # Matriz de rotação horária
            matriz_rotacao = np.array([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=float)

        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_rotacao, np_vetor)

        return vetor_transformado.tolist()

    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='dilatacao_contracao_2d')
def dilatacao_contracao_2d(vetor: list, k: float, is_dilatacao=True):
    '''
    A função 'dilatacao_contracao_2d' realiza uma transformação de escala (dilatação ou contração)
    em um vetor 2D. A dilatação aumenta o vetor e a contração o encolhe.

    Parâmetros:
    - vetor (list): O vetor 2D a ser transformado, por exemplo, [x, y].
    - k (float): O coeficiente de escala. Para dilatação, k > 1. Para contração, 0 < k < 1.
    - is_dilatacao (bool): Se True, a operação é uma dilatação. Se False, é uma contração. O padrão é True.

    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A transformação 2D é definida apenas para vetores 2D
        if np_vetor.shape[0] != 2:
            raise TypeError("A transformação 2D é definida apenas para vetores 2D.")

        # Valida o coeficiente de escala com base no tipo de operação
        if is_dilatacao:
            if k <= 1:
                raise TypeError("Para dilatação, o coeficiente 'k' deve ser maior que 1.")
        else: # Contração
            if not (0 < k < 1):
                raise TypeError("Para contração, o coeficiente 'k' deve estar entre 0 e 1.")

        # Matriz de escala para dilatação ou contração
        matriz_escala = np.array([
            [k, 0],
            [0, k]
        ], dtype=float)

        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_escala, np_vetor)

        return vetor_transformado.tolist()

    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='dilatacao_contracao_3d')
def dilatacao_contracao_3d(vetor: list, k: float, is_dilatacao=True):
    '''
    A função 'dilatacao_contracao_3d' realiza uma transformação de escala (dilatação ou contração)
    em um vetor 3D. A dilatação aumenta o vetor e a contração o encolhe.

    Parâmetros:
    - vetor (list): O vetor 3D a ser transformado, por exemplo, [x, y, z].
    - k (float): O coeficiente de escala. Para dilatação, k > 1. Para contração, 0 < k < 1.
    - is_dilatacao (bool): Se True, a operação é uma dilatação. Se False, é uma contração. O padrão é True.

    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A transformação 3D é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A transformação 3D é definida apenas para vetores 3D.")

        # Valida o coeficiente de escala com base no tipo de operação
        if is_dilatacao:
            if k <= 1:
                raise TypeError("Para dilatação, o coeficiente 'k' deve ser maior que 1.")
        else: # Contração
            if not (0 < k < 1):
                raise TypeError("Para contração, o coeficiente 'k' deve estar entre 0 e 1.")

        # Matriz de escala para dilatação ou contração
        matriz_escala = np.array([
            [k, 0, 0],
            [0, k, 0],
            [0, 0, k]
        ], dtype=float)

        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_escala, np_vetor)

        return vetor_transformado.tolist()

    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='compressao_expansao_2d')
def compressao_expansao_2d(vetor: list, k: float, is_expansao=True):
    '''
    A função 'compressao_expansao_2d' realiza uma transformação de escala (expansão ou compressão)
    em um vetor 2D, onde uma dimensão é expandida e a outra é contraída.

    Parâmetros:
    - vetor (list): O vetor 2D a ser transformado, por exemplo, [x, y].
    - k (float): O coeficiente de escala. Para expansão, k > 1. Para compressão, 0 < k < 1.
    - is_expansao (bool): Se True, a operação é uma expansão. Se False, é uma compressão. O padrão é True.

    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A transformação 2D é definida apenas para vetores 2D
        if np_vetor.shape[0] != 2:
            raise TypeError("A transformação 2D é definida apenas para vetores 2D.")

        # Valida o coeficiente de escala com base no tipo de operação
        if is_expansao:
            if k <= 1:
                raise TypeError("Para expansão, o coeficiente 'k' deve ser maior que 1.")
        else: # Compressão
            if not (0 < k < 1):
                raise TypeError("Para compressão, o coeficiente 'k' deve estar entre 0 e 1.")

        # Matriz de escala para expansão ou compressão
        matriz_escala = np.array([
            [k, 0],
            [0, 1/k]
        ], dtype=float)

        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_escala, np_vetor)

        return vetor_transformado.tolist()

    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='compressao_expansao_3d')
def compressao_expansao_3d(vetor: list, k: float, is_expansao=True):
    '''
    A função 'compressao_expansao_3d' realiza uma transformação de escala (expansão ou compressão)
    em um vetor 3D, onde uma dimensão é expandida e a outra é contraída.

    Parâmetros:
    - vetor (list): O vetor 3D a ser transformado, por exemplo, [x, y, z].
    - k (float): O coeficiente de escala. Para expansão, k > 1. Para compressão, 0 < k < 1.
    - is_expansao (bool): Se True, a operação é uma expansão. Se False, é uma compressão. O padrão é True.

    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A transformação 3D é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A transformação 3D é definida apenas para vetores 3D.")

        # Valida o coeficiente de escala com base no tipo de operação
        if is_expansao:
            if k <= 1:
                raise TypeError("Para expansão, o coeficiente 'k' deve ser maior que 1.")
        else: # Compressão
            if not (0 < k < 1):
                raise TypeError("Para compressão, o coeficiente 'k' deve estar entre 0 e 1.")

        # Matriz de escala para expansão ou compressão
        matriz_escala = np.array([
            [k, 0, 0],
            [0, 1/k, 0],
            [0, 0, 1]
        ], dtype=float)

        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_escala, np_vetor)

        return vetor_transformado.tolist()

    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='cisalhamento_2d')
def cisalhamento_2d(vetor: list, shx=0.0, shy=0.0):
    '''
    A função 'cisalhamento_2d' aplica uma transformação de cisalhamento (shear)
    a um vetor 2D.

    Parâmetros:
    - vetor (list): O vetor 2D a ser transformado, por exemplo, [x, y].
    - shx (float): Fator de cisalhamento em X, que afeta a coordenada Y.  
    - shy (float): Fator de cisalhamento em Y, que afeta a coordenada X.  

    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A transformação 2D é definida apenas para vetores 2D
        if np_vetor.shape[0] != 2:
            raise TypeError("A transformação 2D é definida apenas para vetores 2D.")

        # Matriz de cisalhamento
        matriz_cisalhamento = np.array([
            [1, shx],
            [shy, 1]
        ], dtype=float)

        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_cisalhamento, np_vetor)

        return vetor_transformado.tolist()

    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='cisalhamento_3d')
def cisalhamento_3d(vetor: list, shxy=0.0, shxz=0.0, shyx=0.0, shyz=0.0, shzx=0.0, shzy=0.0):
    '''
    A função 'cisalhamento_3d' aplica uma transformação de cisalhamento (shear)
    a um vetor 3D.

    Parâmetros:
    - vetor (list): O vetor 3D a ser transformado, por exemplo, [x, y, z].
    - shxy (float): Fator de cisalhamento do eixo Y em relação ao eixo X.
    - shxz (float): Fator de cisalhamento do eixo Z em relação ao eixo X.
    - shyx (float): Fator de cisalhamento do eixo X em relação ao eixo Y.
    - shyz (float): Fator de cisalhamento do eixo Z em relação ao eixo Y.
    - shzx (float): Fator de cisalhamento do eixo X em relação ao eixo Z.
    - shzy (float): Fator de cisalhamento do eixo Y em relação ao eixo Z.
    Retorna:
    - list: O vetor transformado.
    - str: Uma mensagem de erro clara se a operação falhar.
    '''
    try:
        # Converte a lista para um array do NumPy e o torna um vetor 1D
        np_vetor = np.array(vetor, dtype=float).flatten()

        # A transformação 3D é definida apenas para vetores 3D
        if np_vetor.shape[0] != 3:
            raise TypeError("A transformação 3D é definida apenas para vetores 3D.")

        # Matriz de cisalhamento
        matriz_cisalhamento = np.array([
            [1, shxy, shxz],
            [shyx, 1, shyz],
            [shzx, shzy, 1]
        ], dtype=float)

        # Aplica a transformação através da multiplicação da matriz pelo vetor
        vetor_transformado = np.dot(matriz_cisalhamento, np_vetor)

        return vetor_transformado.tolist()

    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='projecao_ortogonal_2d_angulo_simples')
def projecao_ortogonal_2d_angulo_simples(vetor: list, angle:float, is_degrees=True):
    """
    Projeta um vetor 2D ortogonalmente sobre uma reta pela origem definida por um ângulo.

    Args:
        vetor (np.array): Vetor 2D a ser projetado.
        angle (float): Ângulo da reta em relação ao eixo X.
        is_degrees (bool): True se o ângulo estiver em graus.

    Returns:
        np.array: Vetor projetado.
    """
    try:
        vetor=np.array(vetor, dtype=float).flatten()
    # --- Verificação de Erros Adicionada ---
        if vetor.shape != (2,):
            raise ValueError("O argumento 'vetor' deve ser um np.array 2D.")


        if is_degrees:
            theta = np.radians(angle)
        else:
            theta = angle

        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Matriz de projeção direta usando apenas o ângulo
        P = np.array([
            [cos_t**2, cos_t*sin_t],
            [cos_t*sin_t, sin_t**2]
        ])
        result=P @ vetor
        return result.tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='reflexao_reta_angulo')
def reflexao_reta_angulo(vetor:list, angulo_graus):
    """
    Realiza a reflexão de um vetor em relação a uma reta que passa pela origem
    e forma um determinado ângulo com o eixo x no espaço 2D.
    
    Args:
        vetor (np.array): O vetor 2D a ser transformado.
        angulo_graus (float): O ângulo da reta de reflexão em graus.
    
    Returns:
        np.array: O vetor transformado.
    """
    try:
        vetor=np.array(vetor, dtype=float).flatten()
        # --- Verificação de Erros Adicionada ---
        if vetor.shape != (2,):
            raise ValueError("O argumento 'vetor' deve ser um np.array 2D.")


        # Converte o ângulo da reta para radianos
        theta = np.radians(angulo_graus)

        # Matriz de reflexão em uma reta com ângulo theta
        matriz_reflexao = np.array([
            [np.cos(2 * theta), np.sin(2 * theta)],
            [np.sin(2 * theta), -np.cos(2 * theta)]
        ])

        vetor_transformado = matriz_reflexao @ vetor
        return vetor_transformado.tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='verificar_injetora')
def verificar_injetora(matriz:list):
    """
    Verifica se uma transformação matricial quadrada é injetora.

    Uma transformação linear de Rn para Rn (matriz quadrada) é injetora
    se, e somente se, o determinante da matriz for diferente de zero.
    
    Args:
        matriz (np.array): A matriz de transformação a ser verificada (deve ser quadrada).
    
    Returns:
        bool: True se a transformação é injetora, False caso contrário.
    """
    try:
        vetor=np.array(vetor, dtype=float).flatten()
        # Verifica se a matriz é quadrada antes de calcular o determinante
        if matriz.shape[0] != matriz.shape[1]:
            # Lança um erro para que a aplicação possa lidar com isso
            raise ValueError("A função 'verificar_injetora' foi projetada para matrizes quadradas (transformações de Rn para Rn).")

        # Calcula o determinante da matriz
        determinante = np.linalg.det(matriz)

        # Usamos uma tolerância para evitar problemas de ponto flutuante
        # A transformação é injetora se o determinante for diferente de zero
        return not np.isclose(determinante, 0)
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'
    
@app.tool(name='encontrar_inversa_matriz')
def encontrar_inversa_matriz(matriz:list):
    """
    Encontra a matriz inversa de uma transformação injetora.

    Uma matriz é invertível se, e somente se, seu determinante for diferente de zero.
    
    Args:
        matriz (np.array): A matriz quadrada de uma transformação injetora.
    
    Returns:
        np.array: A matriz inversa, se existir.
    
    Raises:
        ValueError: Se a matriz não for quadrada ou não for invertível.
    """
    try:
        matriz=np.array(matriz, dtype=float)
        # Verifica se a matriz é quadrada
        if matriz.shape[0] != matriz.shape[1]:
            raise ValueError("A matriz deve ser quadrada para ter uma inversa.")

        # Verifica se a matriz é injetora (determinante != 0)
        determinante = np.linalg.det(matriz)
        if np.isclose(determinante, 0):
            raise ValueError("A matriz não é injetora e, portanto, não possui uma inversa.")

        # Retorna a inversa
        return np.linalg.inv(matriz)
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

def translacao_2d(vetor:list, tx:float, ty:float):
    """
    Realiza a translação de um vetor 2D usando coordenadas homogêneas.
    
    Args:
        vetor (np.array): O vetor 2D a ser transformado, por exemplo, np.array([x, y]).
        tx (float): A quantidade de translação no eixo x.
        ty (float): A quantidade de translação no eixo y.
        
    Returns:
        np.array: O vetor 2D transformado.
        
    Raises:
        ValueError: Se o vetor de entrada não tiver 2 dimensões.
    """
    try:
        vetor=np.array(vetor, dtype=float).flatten()
        # Valida se o vetor de entrada é um vetor 2D
        if vetor.shape != (2,):
            raise ValueError("O vetor de entrada deve ser 2D, com o formato np.array([x, y]).")

        # Converte o vetor 2D para coordenadas homogêneas (3D)
        vetor_homogeneo = np.append(vetor, 1)

        # Matriz de translação 3x3
        matriz_translacao = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

        # Aplica a multiplicação da matriz
        vetor_transformado_homogeneo = matriz_translacao @ vetor_homogeneo

        # Retorna o vetor de volta para o formato 2D
        return vetor_transformado_homogeneo[:2].tolist()
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='translacao_3d')
def translacao_3d(vetor:list, tx:float, ty:float, tz:float):
    """
    Realiza a translação de um vetor 3D usando coordenadas homogêneas.
    
    Args:
        vetor (np.array): O vetor 3D a ser transformado, por exemplo, np.array([x, y, z]).
        tx (float): A quantidade de translação no eixo x.
        ty (float): A quantidade de translação no eixo y.
        tz (float): A quantidade de translação no eixo z.
        
    Returns:
        np.array: O vetor 3D transformado.
        
    Raises:
        ValueError: Se o vetor de entrada não tiver 3 dimensões.
    """
    try:
        vetor=np.array(vetor, dtype=float).flatten()
        # Valida se o vetor de entrada é um vetor 3D
        if vetor.shape != (3,):
            raise ValueError("O vetor de entrada deve ser 3D, com o formato np.array([x, y, z]).")

        # Converte o vetor 3D para coordenadas homogêneas (4D)
        vetor_homogeneo = np.append(vetor, 1)

        # Matriz de translação 4x4
        matriz_translacao = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])

        # Aplica a multiplicação da matriz
        vetor_transformado_homogeneo = matriz_translacao @ vetor_homogeneo

        # Retorna o vetor de volta para o formato 3D
        return vetor_transformado_homogeneo[:3]
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='converter_para_homogenea')
def converter_para_homogenea(vetor):
    """
    Converte um vetor de Rn para coordenadas homogêneas em Rn+1.
    
    Args:
        vetor (np.array): O vetor de n dimensões.
        
    Returns:
        np.array: O vetor em coordenadas homogêneas com a última coordenada igual a 1.
    """
    vetor=np.array(vetor, dtype=float).flatten()
    return np.append(vetor, 1)

@app.tool(name='verificar_espaco_vetorial')
def verificar_espaco_vetorial(conjunto_vetores, vetor_nulo):
    """
    Verifica de forma prática se um conjunto de vetores forma um espaço vetorial,
    testando uma amostra dos dez axiomas principais.

    Esta função não realiza uma prova matemática rigorosa, mas sim uma verificação
    prática para fins de demonstração, usando os vetores fornecidos como exemplos.
    
    Args:
        conjunto_vetores (list ou set): Uma coleção de vetores para teste.
        vetor_nulo (tuple ou np.array): O vetor nulo esperado para o conjunto.

    Returns:
        bool: True se todos os axiomas testados passarem, False caso contrário.
    Raises:
        ValueError: Se um dos axiomas falhar durante a verificação.
    """
    try:
        # 1. Conjunto não vazio?
        if not conjunto_vetores:
            raise ValueError("Erro: O conjunto de vetores não pode ser vazio.")

        print("Iniciando a verificação dos axiomas de espaço vetorial...")

        # Converte os vetores para tuplas para facilitar a comparação em conjuntos
        conjunto_vetores_tuplas = [tuple(v) for v in conjunto_vetores]
        vetor_nulo = np.array(vetor_nulo)

        # --- Axiomas de Adição ---

        # Axioma 1: Fechamento sob a Adição (u, v em V => u+v em V)
        print("-> Verificando fechamento sob a adição...")
        try:
            u = np.array(random.choice(conjunto_vetores))
            v = np.array(random.choice(conjunto_vetores))
            soma = vector_sum(u, v)
            if tuple(soma) not in conjunto_vetores_tuplas:
                raise ValueError(f"Falha no Axioma 1: A soma {soma} não pertence ao conjunto.")
        except IndexError:
            raise ValueError("Erro: O conjunto precisa ter pelo menos 2 vetores para testar a adição.")

        # Axioma 2: Comutatividade (u + v = v + u)
        print("-> Verificando comutatividade da adição...")
        u, v = np.array(random.choice(conjunto_vetores)), np.array(random.choice(conjunto_vetores))
        if not np.array_equal(vector_sum(u, v), vector_sum(v, u)):
            raise ValueError(f"Falha no Axioma 2: {sum(u, v)} != {sum(v, u)}")

        # Axioma 3: Associatividade ((u + v) + w = u + (v + w))
        print("-> Verificando associatividade da adição...")
        if len(conjunto_vetores) > 2:
            u, v, w = np.array(random.choice(conjunto_vetores)), np.array(random.choice(conjunto_vetores)), np.array(random.choice(conjunto_vetores))
            soma_esquerda = vector_sum(vector_sum(u, v), w)
            soma_direita = vector_sum(u, vector_sum(v, w))
            if not np.array_equal(soma_esquerda, soma_direita):
                raise ValueError(f"Falha no Axioma 3: {soma_esquerda} != {soma_direita}")

        # Axioma 4: Existência do Vetor Nulo (u + 0 = u)
        print("-> Verificando a propriedade do vetor nulo...")
        u = np.array(random.choice(conjunto_vetores))
        if not np.array_equal(vector_sum(u, vetor_nulo), u):
            raise ValueError(f"Falha no Axioma 4: A soma de {u} com o vetor nulo não resulta no próprio {u}.")

        # Axioma 5: Existência do Inverso Aditivo (para cada u, existe -u tal que u + (-u) = 0)
        print("-> Verificando a existência do inverso aditivo...")
        u = np.array(random.choice(conjunto_vetores))
        inverso_u = vector_mult_by_an_integer(-1, u)
        if not np.array_equal(vector_sum(u, inverso_u), vetor_nulo):
            raise ValueError(f"Falha no Axioma 5: A soma de {u} e seu inverso aditivo não resulta no vetor nulo.")

        # --- Axiomas de Multiplicação por Escalar ---

        # Axioma 6: Fechamento sob a Multiplicação por Escalar (alpha * u em V)
        print("-> Verificando fechamento sob a multiplicação por escalar...")
        try:
            u = np.array(random.choice(conjunto_vetores))
            escalar = random.uniform(-10, 10)
            produto = vector_mult_by_an_integer(escalar, u)
            if tuple(produto) not in conjunto_vetores_tuplas:
                raise ValueError(f"Falha no Axioma 6: O produto por escalar {produto} não pertence ao conjunto.")
        except IndexError:
            raise ValueError("Erro: O conjunto precisa ter pelo menos 1 vetor para testar a multiplicação por escalar.")

        # Axioma 7: Distributividade sobre a Adição de Vetores (alpha*(u+v) = alpha*u + alpha*v)
        print("-> Verificando distributividade sobre a adição de vetores...")
        alpha = random.uniform(-10, 10)
        u, v = np.array(random.choice(conjunto_vetores)), np.array(random.choice(conjunto_vetores))
        lado_esquerdo = vector_mult_by_an_integer(alpha, vector_sum(u, v))
        lado_direito = vector_sum(vector_mult_by_an_integer(alpha, u), vector_mult_by_an_integer(alpha, v))
        if not np.allclose(lado_esquerdo, lado_direito):
            raise ValueError(f"Falha no Axioma 7: {lado_esquerdo} != {lado_direito}")

        # Axioma 8: Distributividade sobre a Adição de Escalares ((alpha+beta)*u = alpha*u + beta*u)
        print("-> Verificando distributividade sobre a adição de escalares...")
        alpha, beta = random.uniform(-10, 10), random.uniform(-10, 10)
        u = np.array(random.choice(conjunto_vetores))
        lado_esquerdo = vector_mult_by_an_integer(alpha + beta, u)
        lado_direito = vector_sum(vector_mult_by_an_integer(alpha, u), vector_mult_by_an_integer(beta, u))
        if not np.allclose(lado_esquerdo, lado_direito):
            raise ValueError(f"Falha no Axioma 8: {lado_esquerdo} != {lado_direito}")

        # Axioma 9: Associatividade da Multiplicação por Escalar (alpha*(beta*u) = (alpha*beta)*u)
        print("-> Verificando associatividade da multiplicação por escalar...")
        alpha, beta = random.uniform(-10, 10), random.uniform(-10, 10)
        u = np.array(random.choice(conjunto_vetores))
        lado_esquerdo = vector_mult_by_an_integer(alpha, vector_mult_by_an_integer(beta, u))
        lado_direito = vector_mult_by_an_integer(alpha * beta, u)
        if not np.allclose(lado_esquerdo, lado_direito):
            raise ValueError(f"Falha no Axioma 9: {lado_esquerdo} != {lado_direito}")

        # Axioma 10: Identidade Multiplicativa (1*u = u)
        print("-> Verificando identidade multiplicativa...")
        u = np.array(random.choice(conjunto_vetores))
        if not np.array_equal(vector_mult_by_an_integer(1, u), u):
            raise ValueError(f"Falha no Axioma 10: 1 * {u} != {u}")

        print("\nVerificação prática concluída com sucesso! Todos os axiomas testados são válidos.")
        return True
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='verificar_consistencia')
def verificar_consistencia(matriz_coeficientes:list, vetor_resultados:list):
    """
    Verifica se um sistema linear Ax = b é consistente e o classifica.

    A função usa o Teorema de Rouché-Frobenius (comparação de postos) para
    determinar a consistência do sistema. Para sistemas quadrados, ela também
    usa o determinante como um atalho para identificar soluções únicas.

    Args:
        matriz_coeficientes (list): A matriz A do sistema.
        vetor_resultados (list): O vetor b do sistema.

    Returns:
        str: Uma string que descreve o tipo de sistema:
             "Sistema Consistente Determinado" (solução única)
             "Sistema Consistente Indeterminado" (infinitas soluções)
             "Sistema Inconsistente" (sem solução)
    Raises:
        ValueError: Se as dimensões da matriz e do vetor não forem compatíveis.
    """
    try:
        matriz_coeficientes=np.array(matriz_coeficientes, dtype=float)
        vetor_resultados=np.array(vetor_resultados, dtype=float)
        # Garante que a matriz de coeficientes é 2D
        if matriz_coeficientes.ndim != 2:
            raise ValueError("A matriz de coeficientes deve ser uma matriz 2D.")

        # Garante que o vetor de resultados é 1D
        if vetor_resultados.ndim != 1:
            raise ValueError("O vetor de resultados deve ser um vetor 1D.")

        # Verifica se o número de linhas da matriz e do vetor são compatíveis
        if matriz_coeficientes.shape[0] != vetor_resultados.shape[0]:
            raise ValueError("O número de linhas da matriz de coeficientes deve ser igual ao comprimento do vetor de resultados.")

        # Constrói a matriz ampliada [A|b]
        matriz_ampliada = np.hstack((matriz_coeficientes, vetor_resultados.reshape(-1, 1)))

        # Calcula os postos (ranks)
        posto_A = np.linalg.matrix_rank(matriz_coeficientes)
        posto_ampliada = np.linalg.matrix_rank(matriz_ampliada)

        # Obtém o número de incógnitas (número de colunas da matriz de coeficientes)
        num_incognitas = matriz_coeficientes.shape[1]

        # Imprime os postos para fins de depuração e visualização
        print(f"Posto da matriz de coeficientes (A): {posto_A}")
        print(f"Posto da matriz ampliada ([A|b]): {posto_ampliada}")
        print(f"Número de incógnitas: {num_incognitas}")
        print("-" * 30)

        # Classifica o sistema com base nos postos
        if posto_A < posto_ampliada:
            return "Sistema Inconsistente"
        elif posto_A == posto_ampliada:
            if posto_A == num_incognitas:
                return "Sistema Consistente Determinado"
            else:
                return "Sistema Consistente Indeterminado"
        else:
            # Esta condição não deve ocorrer na teoria, mas é uma boa prática
            # retornar um valor em caso de erro.
            return "Erro: Falha na verificação de consistência"
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

# equation_func é a condição que define o subconjunto.
# Por exemplo, para W = {(x,y) em R² ; x - y = 0}, temos:
@app.tool(name='is_subspace')
def is_subspace(vectors:list, eq_add:function, eq_scalar:function):
    """
    Verifica se um conjunto de vetores forma um subespaço vetorial sob as operações definidas.

    Args:
        vectors (list): Lista ou array de vetores para testar.
        eq_add (func): Função que recebe um vetor e retorna True se ele pertence ao conjunto (após soma).
        eq_scalar (func): Função que recebe um vetor e retorna True se ele pertence ao conjunto (após multiplicação por escalar).

    Returns:
        bool: True se o conjunto é um subespaço vetorial, False caso contrário.

    A função verifica:
        - Presença do vetor nulo.
        - Fechamento sob adição.
        - Fechamento sob multiplicação por escalares (testados para alguns valores).
    """
    try:
        # Verifica se o vetor nulo satisfaz ambas as condições
        zero = np.zeros_like(vectors[0])
        if not eq_add(zero) or not eq_scalar(zero):
            return False

        # Verifica fechamento sob adição
        for i in range(len(vectors)):
            for j in range(len(vectors)):
                if not eq_add(vectors[i] + vectors[j]):
                    return False

        # Verifica fechamento sob multiplicação por escalar
        for v in vectors:
            for a in [2, -1, 0.5]:  # alguns escalares de teste
                if not eq_scalar(a * v):
                    return False

        return True
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='eh_combinacao_linear')
def eh_combinacao_linear(vetor_alvo: list,
                         vetores_base: list,
                         tolerancia: float = 1e-10):
    """
    Verifica se um vetor é combinação linear de outros vetores usando uma
    abordagem robusta que funciona para qualquer tipo de sistema (quadrado,
    sobre-determinado ou sub-determinado).

    Parâmetros:
        vetor_alvo (list ou np.array): Vetor que queremos verificar.
        vetores_base (list): Lista dos vetores base.
        tolerancia (float): Tolerância para considerar a solução válida.

    Retorna:
        tuple: (é_combinacao_linear, coeficientes, explicacao)
            - é_combinacao_linear (bool): True se for combinação linear
            - coeficientes (np.array ou None): Coeficientes da combinação linear
            - explicacao (str): Explicação detalhada do resultado
    """
    try:
        # 1. Converter para arrays numpy e fazer verificações iniciais
        vetor_alvo_np = np.array(vetor_alvo, dtype=float)

        # 2. Verifica se a lista de vetores base está vazia
        if not vetores_base:
            if np.isclose(np.linalg.norm(vetor_alvo_np), 0):
                return True, np.array([]), "Vetor nulo é combinação linear de uma base vazia."
            else:
                return False, None, "A base está vazia e o vetor alvo não é o vetor nulo."

        vetores_base_np = [np.array(v, dtype=float) for v in vetores_base]

        # Verifica se todos os vetores têm a mesma dimensão
        dimensao = len(vetor_alvo_np)
        for i, v in enumerate(vetores_base_np):
            if len(v) != dimensao:
                return False, None, f"Erro: O vetor base na posição {i+1} tem dimensão diferente do vetor alvo."

        # 3. Montar a matriz A e a matriz ampliada [A|b]
        A = np.column_stack(vetores_base_np)
        matriz_ampliada = np.hstack((A, vetor_alvo_np.reshape(-1, 1)))

        # 4. Calcular os postos (ranks)
        rank_A = np.linalg.matrix_rank(A, tol=tolerancia)
        rank_ampliada = np.linalg.matrix_rank(matriz_ampliada, tol=tolerancia)

        # 5. Imprimir informações para depuração
        print(f"Dimensão do vetor alvo: {dimensao}")
        print(f"Número de vetores base: {A.shape[1]}")
        print(f"Posto da matriz de bases (A): {rank_A}")
        print(f"Posto da matriz ampliada ([A|b]): {rank_ampliada}")
        print("-" * 40)

        # 6. Checar a consistência
        if rank_A == rank_ampliada:
            # O sistema é consistente (tem solução)
            # Encontrar os coeficientes usando mínimos quadrados
            try:
                # np.linalg.lstsq é mais robusto que np.linalg.solve para sistemas não-quadrados ou singulares
                coeficientes, residuo, posto, s = np.linalg.lstsq(A, vetor_alvo_np, rcond=None)

                explicacao = f"✓ É combinação linear! O sistema é consistente.\n"
                explicacao += f"O vetor alvo está no espaço gerado pelos vetores base.\n"
                explicacao += f"Coeficientes encontrados: {coeficientes}\n"
                # O resíduo (residual) é a norma do erro. Para um sistema consistente, deve ser zero.
                explicacao += f"Erro numérico (residual): {np.linalg.norm(A @ coeficientes - vetor_alvo_np):.2e}"

                return True, coeficientes, explicacao
            except Exception as e:
                return False, None, f"Erro ao resolver o sistema: {str(e)}"
        else:
            # O sistema é inconsistente (não tem solução)
            explicacao = f"✗ Não é combinação linear.\n"
            explicacao += f"Motivo: O posto da matriz de bases ({rank_A}) é diferente do posto da matriz ampliada ({rank_ampliada}).\n"
            explicacao += f"O vetor alvo não está no espaço gerado pelos vetores base."
            return False, None, explicacao
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='posto_matriz_svd')
def posto_matriz_svd(matriz: list, tolerancia: float = 1e-10):
    """
    Calcula o posto (rank) de uma matriz usando SVD.
    """
    try:
        matriz = np.array(matriz, dtype=float)
        _, valores_singulares, _ = np.linalg.svd(matriz)
        return np.sum(valores_singulares > tolerancia)
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

def vetores_geram_espaco(vetores_geradores: list,
                         vetores_teste: list = None,
                         espaco_alvo: str = None,
                         tolerancia: float = 1e-10):
    """
    Verifica se um conjunto de vetores gera um espaço vetorial específico.
    
    Pode verificar:
    1. Se os vetores geram todo o R^n (dimensão n)
    2. Se vetores específicos estão no espaço gerado
    
    Parâmetros:
        vetores_geradores (list): Conjunto de vetores que supostamente geram o espaço.
        vetores_teste (list, opcional): Vetores para testar se estão no espaço gerado.
        espaco_alvo (str, opcional): 'R2', 'R3', etc. para verificar se gera todo o R^n.
        tolerancia (float): Tolerância numérica para comparações.
    
    Retorna:
        tuple: (resultado, informacoes_detalhadas)
    """
    # Converte os vetores de entrada para arrays numpy para facilitar as operações
    try:
        vetores_geradores_np = [np.array(v, dtype=float) for v in vetores_geradores]
    except Exception as e:
        return False, {"erro": f"Erro na conversão dos vetores geradores: {e}"}

    if not vetores_geradores_np:
        return False, {"erro": "A lista de vetores geradores está vazia."}
    
    # Obtém a dimensão dos vetores (assumindo que todos têm a mesma)
    dimensao = len(vetores_geradores_np[0])
    
    # Verifica se todos os vetores têm a mesma dimensão
    for i, v in enumerate(vetores_geradores_np):
        if len(v) != dimensao:
            return False, {"erro": f"O vetor na posição {i} tem uma dimensão diferente ({len(v)} != {dimensao})."}
    
    # Constrói a matriz com os vetores geradores como colunas
    # O posto desta matriz nos dá a dimensão do espaço gerado
    matriz_geradores = np.column_stack(vetores_geradores_np)
    posto_geradores = posto_matriz_svd(matriz_geradores, tolerancia)
    
    informacoes = {
        "dimensao_vetores": dimensao,
        "num_vetores_geradores": len(vetores_geradores_np),
        "posto_geradores": posto_geradores,
        "dimensao_espaco_gerado": posto_geradores,
        "vetores_linearmente_independentes": posto_geradores == len(vetores_geradores_np)
    }
    
    # --- Caso 1: Verificar se os vetores geram todo o R^n ---
    if espaco_alvo:
        try:
            n_esperado = int(espaco_alvo.lower().replace('r', ''))
            if dimensao != n_esperado:
                informacoes.update({
                    "erro": f"A dimensão dos vetores ({dimensao}) não corresponde ao espaço alvo ({espaco_alvo}).",
                    "gera_espaco_completo": False
                })
                return False, informacoes
        except ValueError:
            return False, {"erro": f"O formato do espaço alvo '{espaco_alvo}' é inválido. Use 'R2', 'R3', etc."}

        # Um conjunto de vetores gera R^n se e somente se o posto da matriz formada por eles
        # for igual a 'n' (a dimensão do espaço).
        gera_completo = posto_geradores == dimensao
        informacoes.update({
            "espaco_alvo": espaco_alvo,
            "gera_espaco_completo": gera_completo,
            "dimensoes_faltantes": dimensao - posto_geradores
        })
        
        return gera_completo, informacoes
    
    # --- Caso 2: Verificar se vetores específicos estão no espaço gerado ---
    if vetores_teste:
        try:
            vetores_teste_np = [np.array(v, dtype=float) for v in vetores_teste]
        except Exception as e:
            return False, {"erro": f"Erro na conversão dos vetores de teste: {e}"}
            
        resultados_teste = []
        for i, v_teste in enumerate(vetores_teste_np):
            if len(v_teste) != dimensao:
                resultados_teste.append({
                    "vetor_teste": list(v_teste),
                    "esta_no_espaco": False,
                    "erro": f"O vetor de teste na posição {i} tem uma dimensão incorreta ({len(v_teste)} != {dimensao})."
                })
                continue
            
            # Para verificar se um vetor está no espaço gerado, tentamos resolver o sistema linear
            # A * x = b, onde A é a matriz de geradores e b é o vetor de teste.
            # Usamos np.linalg.lstsq para uma solução numérica robusta.
            try:
                # O método retorna a solução (coeficientes), resíduos, posto e valores singulares.
                coeficientes, residuo, _, _ = np.linalg.lstsq(matriz_geradores, v_teste, rcond=None)
                
                # O vetor está no espaço se o resíduo da solução for próximo de zero.
                # A norma do resíduo é a medida do erro.
                esta_no_espaco = np.isclose(np.linalg.norm(v_teste - matriz_geradores @ coeficientes), 0, atol=tolerancia)
                
                resultados_teste.append({
                    "vetor_teste": list(v_teste),
                    "esta_no_espaco": bool(esta_no_espaco),
                    "coeficientes": list(coeficientes),
                    "erro_residual": np.linalg.norm(v_teste - matriz_geradores @ coeficientes)
                })
            except Exception as e:
                resultados_teste.append({
                    "vetor_teste": list(v_teste),
                    "esta_no_espaco": False,
                    "erro": f"Não foi possível resolver o sistema linear para este vetor: {e}"
                })
        
        informacoes["testes_vetores"] = resultados_teste
        
        # Retorna True se TODOS os vetores de teste estiverem no espaço
        todos_no_espaco = all(r["esta_no_espaco"] for r in resultados_teste)
        return todos_no_espaco, informacoes
    
    # --- Caso 3: Apenas análise dos vetores geradores ---
    # Se nem espaco_alvo nem vetores_teste foram fornecidos, apenas retorna as informações sobre os vetores geradores.
    return True, informacoes

@app.tool(name='check_linear_independence')
def check_linear_independence(vectors: list, dim: int):
    """
    Verifica se um conjunto de vetores em R^dim é linearmente independente ou dependente.
    
    A verificação é baseada no posto (rank) da matriz formada pelos vetores.
    Um conjunto de vetores é linearmente independente se e somente se o posto
    da matriz for igual ao número de vetores no conjunto.
    
    Parameters:
        vectors (list): Lista de vetores, onde cada vetor é uma lista ou array numérico.
        dim (int): Dimensão do espaço vetorial (ex: 2 para R², 3 para R³).
    
    Returns:
        str: "Independente" ou "Dependente".
    """
    try:
        # Verificação de entrada
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("A dimensão deve ser um inteiro positivo (ex: 2 para R²).")

        if not isinstance(vectors, list) or len(vectors) == 0:
            raise ValueError("A lista de vetores não pode estar vazia e deve ser do tipo list.")

        # Garantir que todos os vetores sejam arrays numpy e com a dimensão correta
        formatted_vectors = []
        for i, v in enumerate(vectors):
            v = np.array(v)  # converte para np.array caso seja lista/tupla
            if v.shape[0] != dim:
                raise ValueError(f"O vetor {i+1} não tem dimensão {dim}. Recebido: dimensão {v.shape[0]}.")
            formatted_vectors.append(v)

        num_vectors = len(formatted_vectors)

        # Caso tenha mais vetores que a dimensão, já é dependente
        if num_vectors > dim:
            return "Dependente (mais vetores que a dimensão do espaço)"

        # Cria matriz com vetores como colunas
        matrix = np.column_stack(formatted_vectors)

        # Calcula posto
        rank = np.linalg.matrix_rank(matrix)

        if rank == num_vectors:
            return "Independente"
        else:
            return "Dependente"
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'
    

@app.tool(name='is_basis')
def is_basis(vectors: list, dim: int):
    """
    Verifica se um conjunto de vetores forma uma base de R^dim.
    
    Um conjunto de vetores é uma base se satisfaz duas condições:
    1. O número de vetores no conjunto é igual à dimensão do espaço.
    2. Os vetores são linearmente independentes.
    
    Parameters:
        vectors (list): Lista de vetores, onde cada vetor é uma lista ou array numérico.
        dim (int): Dimensão do espaço vetorial (ex: 2 para R², 3 para R³).
    
    Returns:
        str: "É base" ou "Não é base" com o motivo.
    """
    try:
        num_vectors = len(vectors)

        if num_vectors < dim:
            return "Não é base (menos vetores que a dimensão)"
        elif num_vectors > dim:
            return "Não é base (mais vetores que a dimensão)"
        else:
            status = check_linear_independence(vectors, dim)
            if status == "Independente":
                return "É base"
            else:
                return "Não é base (vetores linearmente dependentes)"
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='nullspace_basis')
def nullspace_basis(A:list, tol: float = 1e-12):
    """
    Retorna uma base para o núcleo (null space) de uma matriz A.
    
    O núcleo é o conjunto de todos os vetores 'x' que satisfazem a equação matricial Ax = 0.
    A função usa Decomposição por Valor Singular (SVD) para encontrar a base de forma
    numericamente estável. Os vetores de base correspondem às colunas de V que
    possuem valores singulares (S) próximos de zero.
    
    Parameters:
        A (list): A matriz m×n em formato de lista aninhada.
        tol (float): Tolerância numérica para considerar um valor singular como zero.
    
    Returns:
        list[np.ndarray]: Uma lista de vetores que formam a base do núcleo.
                          Se o núcleo for apenas o vetor zero, retorna uma lista vazia.
    """
    A = np.atleast_2d(np.array(A, dtype=float))
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    # rank ≈ quantidade de S > tol
    rank = int(np.sum(S > tol))
    # Linhas de Vt a partir de 'rank' geram o núcleo (cada linha é um vetor base transposto)
    Vt_null = Vt[rank:]
    return [row for row in Vt_null]

@app.tool(name='subspace_basis_from_equations')
def subspace_basis_from_equations(A: list, tol: float = 1e-12):
    """
    Retorna a base, a dimensão e o posto (rank) do núcleo de uma matriz.
    
    É uma função de conveniência que combina o cálculo da base do núcleo
    com a determinação da dimensão e do posto da matriz.
    
    Parameters:
        A (list): A matriz m×n em formato de lista aninhada.
        tol (float): Tolerância numérica para singular values próximos de zero.
    
    Returns:
        tuple: (base, dim, rank)
            - base (list[np.ndarray]): Base do núcleo.
            - dim (int): Dimensão do núcleo (nulidade).
            - rank (int): Posto da matriz.
    """
    try:
        A = np.atleast_2d(np.array(A, dtype=float))
        base = nullspace_basis(A, tol=tol)
        dim = len(base)
        rank = np.linalg.matrix_rank(A)
        return base, dim, rank
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='basis_of_span')
def basis_of_span(vectors, dim: int, tol: float = 1e-12):
    """
    Retorna uma base para o espaço gerado (span) por um conjunto de vetores.
    
    - Se os vetores de entrada forem linearmente independentes (LI), a função
      retorna os próprios vetores (como cópias).
    - Se os vetores forem linearmente dependentes (LD), a função utiliza a
      Decomposição por Valor Singular (SVD) para encontrar uma base ortonormal
      para o espaço-coluna da matriz formada pelos vetores.
    
    Parameters:
        vectors (list): Lista de vetores, onde cada vetor é uma lista ou array.
        dim (int): A dimensão do espaço vetorial dos vetores.
        tol (float): Tolerância para singular values próximos de zero.
    
    Returns:
        list[np.ndarray]: Uma lista de vetores que formam a base do span.
    """
    try:
        # Validação e montagem
        cols = []
        for i, v in enumerate(vectors):
            v = np.array(v, dtype=float)
            if np.iscomplexobj(v):
                raise ValueError("A função só aceita vetores reais.")
            if v.shape[0] != dim:
                raise ValueError(f"Vetor {i+1} tem dimensão {v.shape[0]} ≠ {dim}.")
            cols.append(v)

        if len(cols) == 0:
            raise ValueError("Forneça ao menos um vetor.")

        M = np.column_stack(cols)  # n×k
        rank = np.linalg.matrix_rank(M, tol)
        k = M.shape[1]

        if rank == k:
            return [c.copy() for c in cols]

        # LD: usa SVD para obter base ortonormal
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        r = int(np.sum(S > tol))
        base = [U[:, i].copy() for i in range(r)]
        return base
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='in_subspace')
def in_subspace(v, A, tol: float = 1e-10):
    """
    Checa se um vetor 'v' pertence ao núcleo (null space) de uma matriz 'A'.
    
    O núcleo de A é o conjunto de todos os vetores x tal que Ax = 0.
    A função verifica se o produto da matriz A pelo vetor v é aproximadamente
    igual ao vetor nulo, usando uma tolerância para lidar com imprecisões numéricas.
    
    Parameters:
        v (list): O vetor a ser checado.
        A (list): A matriz.
        tol (float): Tolerância numérica.
    
    Returns:
        bool: True se o vetor está no núcleo, False caso contrário.
    """
    try:
        A = np.atleast_2d(np.array(A, dtype=float))
        v = np.array(v, dtype=float)
        return np.linalg.norm(A @ v) <= tol
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='dimensao_espaco_vetorial')
def dimensao_espaco_vetorial(vectors: list, 
                            espaco_ambiente_dim: int,
                            tol: float = 1e-12):
    """
    Calcula a dimensão de um espaço vetorial V seguindo a definição:
    "O número de elementos (cardinalidade) de uma base B do espaço vetorial V 
    é denominado dimensão do espaço vetorial V."
    
    A função encontra uma base do espaço gerado pelos vetores fornecidos
    e retorna a cardinalidade (número de elementos) dessa base.
    
    Parâmetros:
        vectors (list): Lista de vetores que geram o espaço vetorial V
        espaco_ambiente_dim (int): Dimensão do espaço ambiente (ex: 2 para R², 3 para R³)
        tol (float): Tolerância numérica
    
    Retorna:
        tuple: (dimensao, informacoes_detalhadas)
            - dimensao (int): dim(V) = cardinalidade da base
            - informacoes (dict): detalhes sobre a base e o processo
    """
    try:
        if not vectors:
            return 0, {"erro": "Lista de vetores vazia", "base": [], "eh_espaco_trivial": True}

        # Converter vetores para arrays numpy
        vectors_np = []
        for i, v in enumerate(vectors):
            v_array = np.array(v, dtype=float)
            if len(v_array) != espaco_ambiente_dim:
                raise ValueError(f"Vetor {i+1} tem dimensão {len(v_array)} ≠ {espaco_ambiente_dim}")
            vectors_np.append(v_array)

        # Construir matriz com vetores como colunas
        M = np.column_stack(vectors_np)

        # Calcular posto da matriz (= dimensão do espaço gerado)
        posto = np.linalg.matrix_rank(M, tol)

        # Encontrar uma base usando SVD
        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # Número de valores singulares significativos = dimensão
        dimensao = int(np.sum(S > tol))

        # Extrair base (primeiras 'dimensao' colunas de U)
        if dimensao == 0:
            base = []
        else:
            base = [U[:, i].copy() for i in range(dimensao)]

        # Verificar se os vetores originais já formam uma base
        vetores_originais_sao_base = (dimensao == len(vectors_np) and posto == len(vectors_np))

        # Informações detalhadas
        informacoes = {
            "dimensao": dimensao,
            "cardinalidade_base": dimensao,  # Pela definição: dim(V) = |B|
            "base_ortonormal": base,
            "posto_matriz": posto,
            "num_vetores_originais": len(vectors_np),
            "vetores_originais_sao_base": vetores_originais_sao_base,
            "valores_singulares": list(S),
            "eh_espaco_trivial": dimensao == 0,
            "eh_subespaco_proprio": dimensao < espaco_ambiente_dim,
            "gera_espaco_completo": dimensao == espaco_ambiente_dim,
            "vetores_linearmente_independentes": posto == len(vectors_np)
        }

        return dimensao, informacoes
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='complete_to_basis')
def complete_to_basis(vectors, dim):
    """
    Completa um conjunto de vetores linearmente independentes
    para formar uma base de R^dim.

    Parameters:
        vectors (list of array-like): Vetores independentes dados
        dim (int): Dimensão do espaço vetorial (R^dim)

    Returns:
        list of np.array: Base completa de R^dim
    """
    try:
        # Verificação de entrada
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("A dimensão deve ser um inteiro positivo (ex: 2 para R²).")

        formatted_vectors = [np.array(v, dtype=float) for v in vectors]
        for i, v in enumerate(formatted_vectors):
            if v.shape[0] != dim:
                raise ValueError(f"O vetor {i+1} não tem dimensão {dim}. Recebido: {v.shape[0]}.")

        # Matriz com vetores dados como colunas
        matrix = np.column_stack(formatted_vectors) if formatted_vectors else np.zeros((dim,0))

        # Verifica independência inicial
        if np.linalg.matrix_rank(matrix) != len(formatted_vectors):
            raise ValueError("Os vetores fornecidos não são linearmente independentes.")

        # Se já é base, retorna
        if len(formatted_vectors) == dim:
            return formatted_vectors

        # Tenta adicionar vetores da base canônica até completar a base
        for i in range(dim):
            candidate = np.eye(dim)[:, i]  # vetor canônico
            test_matrix = np.column_stack([matrix, candidate])
            if np.linalg.matrix_rank(test_matrix) > np.linalg.matrix_rank(matrix):
                formatted_vectors.append(candidate)
                matrix = test_matrix
            if len(formatted_vectors) == dim:
                break

        return formatted_vectors
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return f'Erro inesperado: {e}'

@app.tool(name='coordenadas_na_base')
def coordenadas_na_base(v, base):
    """
    Calcula as coordenadas do vetor v em relação a uma base ordenada.

    Parâmetros:
        v (array-like): vetor a ser representado.
        base (list of array-like): base ordenada do espaço vetorial.

    Retorna:
        np.array: coordenadas de v em relação à base.
    """
    try:
        v = np.array(v, dtype=float)
        B = np.column_stack([np.array(b, dtype=float) for b in base])

        n = B.shape[0]
        if B.shape[1] != n:
            raise TypeError("A base não é completa (não tem n vetores em R^n).")

        if np.linalg.matrix_rank(B) != n:
            raise TypeError("Os vetores fornecidos não são linearmente independentes, portanto não formam uma base.")

        # Resolver B * λ = v
        lambdas = np.linalg.solve(B, v)
        return lambdas
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        return f"Erro inesperado: {str(e)}"

@app.tool(name='hits_algorithm')
def hits_algorithm(A:list, tol=1e-4, max_iter=1000):
    """
    Implementa o algoritmo HITS (Hyperlink-Induced Topic Search).
    
    Parâmetros:
        A (list): Matriz de adjacência (n x n).
        tol (float): Critério de parada (tolerância).
        max_iter (int): Número máximo de iterações.
    
    Retorna:
        dict: Resultados contendo número de iterações, solução (vetor autoridade)
              e forma decrescente dos valores.
    """
    try:
        # Verificação de entrada
        if not isinstance(A, list):
            raise ValueError("A matriz A deve ser uma lista")
        if A.shape[0] != A.shape[1]:
            raise ValueError("A matriz de adjacência deve ser quadrada.")

        # Vetor autoridade inicial (grau de entrada)
        a0 = np.sum(A, axis=0).astype(float)

        for i in range(1, max_iter + 1):
            u = A @ a0
            r = np.linalg.norm(u)
            hn = u / r if r != 0 else u

            v = A.T @ hn
            s = np.linalg.norm(v)
            an = v / s if s != 0 else v

            erro = np.abs(an - a0)

            if np.max(erro) <= tol:
                solucao = an
                break
            else:
                a0 = an

        R = np.sort(an)[::-1]

        return {
            "iteracoes": i,
            "solucao": solucao,
            "ordenado_decrescente": R
        }
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        return f"Erro inesperado: {str(e)}"

@app.tool(name='characteristic_polynomial')
def characteristic_polynomial(A:list):
    """
    Calcula os coeficientes do polinômio característico de uma matriz quadrada A.
    Retorna os coeficientes em ordem decrescente de λ.
    """
    try:
        A = np.array(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A matriz deve ser quadrada.")

        # Coeficientes do polinômio característico usando np.poly
        coeffs = np.poly(A)  
        return coeffs
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        return f"Erro inesperado: {str(e)}"

@app.tool(name='eigenvalues_from_poly')
def eigenvalues_from_poly(coeffs:list):
    """
    Dado os coeficientes do polinômio característico, encontra os autovalores.
    Inclui tratamento de erros para garantir que a entrada seja válida.
    
    Args:
        coeffs (list): Uma lista ou array NumPy
        contendo os coeficientes do polinômio em ordem decrescente de potência.
        
    Returns:
        np.ndarray: Um array contendo os autovalores (raízes) do polinômio.
        Retorna None se a entrada for inválida.
    """
    try:
        # A função np.roots lida com a lógica principal,
        # encontrando as raízes do polinômio.
        return np.roots(coeffs)
    except Exception as e:
        return f'Erro:{e}'

@app.tool(name='eigenvectors')
def eigenvectors(A, eigenvals, tol=1e-12):
    """
    Encontra autovetores de uma matriz A dados os autovalores fornecidos.
    Esta versão utiliza a função otimizada null_space da biblioteca SciPy,
    garantindo maior robustez e eficiência.

    Parâmetros:
        A (list ou array): Matriz quadrada.
        eigenvals (list): Lista de autovalores.
        tol (float): Tolerância numérica para considerar valores singulares como zero.

    Retorna:
        dict: Um dicionário onde as chaves são os autovalores e os valores
              são os autovetores associados. Retorna um dicionário com
              uma chave de erro se a entrada for inválida.
    """
    try:
        try:
            A = np.array(A, dtype=float)
            eigenvals = np.array(eigenvals, dtype=float)
        except Exception as e:
            return {"erro": f"Erro ao converter entradas: {e}"}

        # Verifica se A é quadrada
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return {"erro": "A matriz A deve ser quadrada."}

        n = A.shape[0]
        resultado = {}

        for lam in eigenvals:
            M = A - lam * np.eye(n)

            try:
                # Calcula base do núcleo usando scipy (robusta e otimizada)
                base = null_space(M, rcond=tol)  # Cada coluna é um vetor da base
            except Exception as e:
                resultado[lam] = {"erro": f"Erro ao calcular null space: {e}"}
                continue

            if base.shape[1] == 0:
                resultado[lam] = {"autovetores": [], "observacao": "Nenhum autovetor encontrado (núcleo trivial)."}
            else:
                # Cada vetor já vem normalizado da null_space (norma 1)
                resultado[lam] = {"autovetores": base}

        return resultado
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        return f"Erro inesperado: {str(e)}"

@app.tool(name='diagonalizar_matriz_robusta')
def diagonalizar_matriz_robusta(A):
    """
    Diagonaliza uma matriz quadrada A ∈ C^{nxn}, se possível.

    Retorna D, P tais que A = P D P⁻¹.

    Retorna:
        {
            "D": matriz diagonal dos autovalores,
            "P": matriz de autovetores (colunas),
            "autovalores": vetor,
            "eh_diagonalizavel": bool,
            "sobre_reais": bool,
            "P_inv": matriz inversa de P (opcional),
        }
    """
    try:
        A = np.array(A, dtype=complex)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return {"erro": "Matriz deve ser quadrada."}

        try:
            autovalores, P = np.linalg.eig(A)
        except Exception as e:
            return {"erro": f"Erro ao calcular autovalores/autovetores: {e}"}

        n = A.shape[0]
        rank_P = np.linalg.matrix_rank(P)

        if rank_P < n:
            return {
                "eh_diagonalizavel": False,
                "erro": "Autovetores linearmente dependentes. Matriz não é diagonalizável."
            }

        D = np.diag(autovalores)
        P_inv = np.linalg.inv(P)

        return {
            "D": D,
            "P": P,
            "P_inv": P_inv,
            "autovalores": autovalores,
            "eh_diagonalizavel": True,
            "sobre_reais": np.all(np.isreal(autovalores)) and np.all(np.isreal(P))
        }
    except (ValueError, TypeError) as e:
        # Captura erros se a entrada não puder ser convertida para um array de floats
        return f'Erro: {e}'
    except Exception as e:
        return f"Erro inesperado: {str(e)}"
    
@app.tool(name='calcular_potencia_matriz_semelhanca')
def calcular_potencia_matriz_semelhanca(A:list, k:int):
    """
    Calcula a k-ésima potência de uma matriz quadrada A usando diagonalização.

    A função usa a fórmula A^k = P * D^k * P⁻¹, onde D é a matriz diagonal
    e P é a matriz de passagem.

    Args:
        A (list): A matriz quadrada a ser elevada à potência.
        k (int): O expoente.

    Returns:
        list: A matriz resultante A^k, ou um dicionário com uma mensagem
                    de erro se a matriz não for diagonalizável.
    """
    print(f"\nCalculando a potência A^{k}...")
    
    if not isinstance(k, int):
        return {"erro": "O expoente k deve ser um número inteiro."}
    if k < 0:
    # Verificar se todos os autovalores são não-nulos
        if np.any(np.isclose(resultado_diag['autovalores'], 0)):
            return {"erro": "Não é possível calcular potência negativa: autovalores nulos."}
    # Tenta diagonalizar a matriz A e armazena o resultado no dicionário.
    resultado_diag = diagonalizar_matriz_robusta(A)
    
    # Verifica se a diagonalização foi bem-sucedida.
    if not resultado_diag.get("eh_diagonalizavel", False):
        print(f"Não é possível calcular a potência usando semelhança: {resultado_diag.get('erro', 'Matriz não diagonalizável.')}")
        return {"erro": resultado_diag.get('erro', 'Matriz não diagonalizável.')}
    
    # Extrai as matrizes necessárias do dicionário de resultados.
    D = resultado_diag['D']
    P = resultado_diag['P']
    P_inv = resultado_diag['P_inv']
    
    # Calcula a potência da matriz diagonal D.
    try:
        D_k = np.linalg.matrix_power(D, k)
    except np.linalg.LinAlgError:
        return {"erro": "Não foi possível calcular a potência da matriz diagonal."}
    
    # Calcula a potência final A^k = P * D^k * P⁻¹.
    A_k = P @ D_k @ P_inv
    
    print("Cálculo da potência A^k bem-sucedido.")
    return A_k.tolist()

if __name__=="__main__":
    app.run(transport='stdio')