#from fastmcp import FastMCP
from mcp.server.fastmcp import FastMCP
import numpy as np

app=FastMCP('operadores de matrizes')

@app.tool(name='ping')
def ping():
    '''Caso o usuário escrever ping, retornará pong, isso significará ao usuário que o servidor MCP está funcionando e retornando o que foi pedido'''
    return 'pong'


if __name__=="__main__":
    app.run(transport='stdio')