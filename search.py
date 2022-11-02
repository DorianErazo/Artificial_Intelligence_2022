# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).

-----------------------------------------------------------
En esta parte del codigo estaran los algoritmos de busqueda!
-----------------------------------------------------------
"""


import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        #self.start_state
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    -------------------- DFS --------------------
    Visitamos todos los nodos, empezamos por el de mas izquierda en profundidad.
    DFS(grafo, origen, objetivo)
    Esto funciona en stack (LIFO), por lo tanto del util.py utilizaremos la class stack
    El coste puede ser O(v^2)

    """
    "*** YOUR CODE HERE ***"

    #Declaramos la pila con la que vamos a trabajar
    stack = util.Stack()
    #Lista de nodos auxiliares donde añadiremos el path mas <<optimo>>
    visited_nodes = []

    #declare first node
    s0_node = problem.getStartState()
    print("Start:", problem.getStartState())

    "Remember that a search node must contain not only a state but also the information necessary to reconstruct the path (plan) which gets to that state."
    #Pasamos el primer nodo del arbol al stack
    stack.push((s0_node, []))

    while not stack.isEmpty():
        #esto funciona tal que asi; stack [1,2]. Queremos las dos posiciones, la primera y la 2na (donde actuamos)
        #current_node = 1 y coor_nodes = 2. Problem nos devuelve dos datos asi que esto nos vendra bien para todas 
        #los demas apartados!
        current_node, coor_nodes = stack.pop()

        if current_node not in visited_nodes:
            visited_nodes.append(current_node)
            print("Is the start a goal?", problem.isGoalState(problem.getStartState())) 

            if problem.isGoalState(current_node):
                print("action: ", coor_nodes)
                print("Is the start a goal en goal state?", problem.isGoalState(current_node))
                return coor_nodes

            #pasamos 3 condiciones a cumplir xq la funcion getsuccesor pasa 3 atributos
            for next_node, action, cost in problem.getSuccessors(current_node):
                next_actions = coor_nodes + [action]
                stack.push((next_node, next_actions))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    #El problema es basicamente el mismo que el anterior, entonces solo va a cambiar el modo de uso
    #del tipo de dato, que en este caso es una lista.
    #todo el codigo va a ser igual. 
    "*** YOUR CODE HERE ***"

    qlist = util.Queue()
    visited_nodes = []

    #no hace falta añadir el primer nodo a la lista de visitados. Lo hara el While de abajo
    #Aun que descomentemos las lineas de abajo el codigo funcionara con los comandos
    #python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
    #No obstante para que funcione el comando python eightpuzzle.py tenemos que quitar el primer
    #nodo visitado, esto se debe al rango de numeros que acepta el puzzle

    #visited_nodes.append((problem.getStartState(), []))
    #print("vitied nodes al 0", visited_nodes)

    qlist.push((problem.getStartState(), []))
    #print("qlist: ", qlist)

    while not qlist.isEmpty():
        current_node, coord = qlist.pop()

        if current_node not in visited_nodes:
            visited_nodes.append(current_node)

            if problem.isGoalState(current_node):
                return coord

            for next_node, action, cost in problem.getSuccessors(current_node):
                next_action = coord + [action]
                qlist.push((next_node, next_action))
        


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #Declaramos priority queue
    pqlist = util.PriorityQueue()
    visited_nodes = []

    #Pasamos el primer nodo con coste 0 a la cola.
    pqlist.push((problem.getStartState(), [], 0), 0)

    while not pqlist.isEmpty():
        current_node, coord, cost = pqlist.pop()

        if current_node not in visited_nodes:
            visited_nodes.append(current_node)

            if problem.isGoalState(current_node):
               return coord

            #Añadimos los nuevos nodos junto a su coste a la priority queue, de forma que se ordenen de menor a mayor coste.
            for next_node, action, nextcost in problem.getSuccessors(current_node):
                if next_node not in visited_nodes:
                    next_action = coord + [action]
                    priority = cost + nextcost
                    pqlist.push((next_node, next_action, priority), priority)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #https://www.pythonpool.com/a-star-algorithm-python/
    #https://www.youtube.com/watch?v=FQsHjAuVkIs
    """
    Para poder usar el A* vamos a necesitar dos cosas principalmente 
    Heuristica h(n)
    Costo de "caminos" h(n)

    f(n) = g(n) + h(n)

    Para almacenar los datos necesitaremos varias listas
    open_list, guarda el camino de los nodos no visitados (no lo usare)
    closed_list, que guarda el camino de los si visitados

    Otras listas para guarda la distancia del camino 
    path_len (lo tenemos por funcion)
    
    Y nuestros nodos.
    En esta funcion vamos a tener 3 datos que nos llegan desde la llamada de la
    funcion, dos en problem y otro en heuristic.

    (coordenada, accion del nodo actual, coste del actual nodo) y a parte su prioridad (priority queue)
    Usaremos priority queue de la clase Utils ya que asi se nos organizara por
    valor, por ejemplo ti tenemos una lista de numeros la forma de organizar seria de menor a mayor.
    """
    closed_list = []
    
    p_queue = util.PriorityQueue()
    #Al push se le pueden añadir datos tal que: push(self, item, priority). Siendo item lo del parentesis y 0
    #la prioridad
    p_queue.push((problem.getStartState(), [], 0), 0)

    if problem.isGoalState(problem.getStartState()):
        return []

    p_queue = util.PriorityQueue()
    #push puede recibir estos datos push(self, item, priority). lo de los parentesis es item y el 0 
    #es la prioridad
    p_queue.push((problem.getStartState(), [], 0), 0)

    if problem.isGoalState(problem.getStartState()):
        return []

    while not p_queue.isEmpty():

        current_node, coord, current_node_cost = p_queue.pop()

        if current_node not in closed_list:
            closed_list.append(current_node)

            if problem.isGoalState(current_node):
                return coord

            for next_node, action, cost in problem.getSuccessors(current_node):
                next_action = coord + [action]
                #El costo lo sumaremos de esta forma
                new_cost_node = current_node_cost + cost
                #Aqui aplicamos la formula f(n) = g(n)+h(n)
                #nota, usamos heuristic de la abreviacion en la llamada de la funcion en lugar de
                #nullHeuristic ya que si no el autograder nos da un 0. 
                cost_heuristic = new_cost_node + heuristic(next_node,problem)
                p_queue.push((next_node, next_action, new_cost_node),cost_heuristic)



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
