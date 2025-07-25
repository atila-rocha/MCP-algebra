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
                if column>line and A[line][column]!=0:
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
                    if summin < min:
                        min=mins[listmin][summin]
                    if listmin==1:
                        axis=listmin
                    pos=summin
            sum=0
            for line in range(0,A.shape[0]):
                if axis==0:
                    if pos==line:
                        A=A[line]
                        for element in A:
                            sum+= element * pow(-1, line+column) * minor_entrance
                for column in range(0,A.shape[1]):
                    if axis==0:
                        if pos==line:
                            sum+= A[line][column] * pow(-1, line+column) * minor_entrance(A, line, column)
                            #todo
                        pass
            
        else: raise Exception('Não foi possível calcular: A Matriz não é quadrada')
    except Exception as e: return e.args 

@app.tool(name='minor_entrance')
def minor_entrance(A:list, line: int, column: int):
    A=np.array(A)
    #A=np.delete(A, line, axis=0)
    #A=np.delete(A, column, axis=1)
    A_new = A[np.arange(A.shape[0]) != line][:, np.arange(A.shape[1]) != column]

    if A_new.shape==(1,1):
        return A[0][0]
    else: la_place(A)


if __name__=="__main__":
    app.run(transport='stdio')