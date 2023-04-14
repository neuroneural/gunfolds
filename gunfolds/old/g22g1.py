from copy import deepcopy

all_acceptable = []


def find_next_graph(graph_g):
    next_graph = {}
    for i in graph_g:
        next_graph[i] = {}
    for i in graph_g:
        for j in graph_g[i]:
            for m in graph_g[j]:
                if not m in next_graph[i]:
                    next_graph[i][m] = set([(0, 1)])
                elif next_graph[i][m] == set([(0, 2)]) or next_graph[i][m] == set([(0, 3)]):
                    next_graph[i][m] = set([(0, 3)])
            for n in graph_g[i]:
                if j != n:
                    if not n in next_graph[j]:
                        next_graph[j][n] = set([(0, 2)])
                    elif next_graph[j][n] == set([(0, 1)]) or next_graph[j][n] == set([(0, 3)]):
                        next_graph[j][n] = set([(0, 3)])
    return next_graph


def compare(g_two_star, graph_g):
    for i in g_two_star:
        for j in g_two_star[i]:
            if j not in graph_g[i]:
                return False
            elif g_two_star[i][j] != graph_g[i][j] and graph_g[i][j] != set([(0, 3)]):
                return False
    return True


def try_next_double_edge(g_de, g_one_star_de, stack_de):
    global all_acceptable
    if stack_de != []:
        j = stack_de.pop()
        i = stack_de.pop()
        for k in g_one_star_de:
            boolean_first_edge_has_value = False
            boolean_second_edge_has_value = False
            if i in g_one_star_de[k]:
                boolean_first_edge_has_value = True
            if j in g_one_star_de[k]:
                boolean_second_edge_has_value = True
            g_one_star_de[k][i] = 1
            g_one_star_de[k][j] = 1
            g_two_star_de = find_next_graph(g_one_star_de)
            if compare(g_two_star_de, g_de):
                try_next_double_edge(g_de, g_one_star_de, stack_de)
            if not boolean_first_edge_has_value:
                del g_one_star_de[k][i]
            if not boolean_second_edge_has_value:
                del g_one_star_de[k][j]
        stack_de.append(i)
        stack_de.append(j)
    else:
        in_all_accept = False
        for xxxx in all_acceptable:
            if(xxxx == g_one_star_de):
                in_all_accept = True
        if (not in_all_accept):
            all_acceptable.append(deepcopy(g_one_star_de))
            print "Added 1."
# print g_one_star_de
# if g_one_star_de not in all_acceptable:
# all_acceptable.append(g_one_star_de)
        # print "G2 --------------------------"
        # print find_next_graph(g_one_star_de) == g_de
        # print "End-------------------------"


def sample_g_double_edge(g_de, g_one_star_de):
    stack_de = []
    for i in g_de:
        for j in g_de[i]:
            if g_de[i][j] != set([(0, 1)]):
                if int(i) < int(j):
                    stack_de.append(i)
                    stack_de.append(j)
    try_next_double_edge(g_de, g_one_star_de, stack_de)


def try_next_single_edge(g_se, g_one_star_se, stack_se):
    if stack_se != []:
        j = stack_se.pop()
        i = stack_se.pop()
        for k in g_one_star_se:
            boolean_first_edge_has_value = False
            boolean_second_edge_has_value = False
            if k in g_one_star_se[i]:
                boolean_first_edge_has_value = True
            if j in g_one_star_se[k]:
                boolean_second_edge_has_value = True
            g_one_star_se[i][k] = 1
            g_one_star_se[k][j] = 1
            g_two_star_se = find_next_graph(g_one_star_se)
            if compare(g_two_star_se, g_se):
                try_next_single_edge(g_se, g_one_star_se, stack_se)
            if not boolean_first_edge_has_value:
                del g_one_star_se[i][k]
            if not boolean_second_edge_has_value and j in g_one_star_se[k]:
                del g_one_star_se[k][j]
        stack_se.append(i)
        stack_se.append(j)
    else:
        sample_g_double_edge(g_se, g_one_star_se)


def sample_g_single_edge(g_se, g_one_star_se):
    stack_se = []
    for i in g_se:
        for j in g_se[i]:
            if g_se[i][j] != set([(0, 2)]):
                stack_se.append(i)
                stack_se.append(j)
    try_next_single_edge(g_se, g_one_star_se, stack_se)


def main(gStart):
    global all_acceptable
    all_acceptable = []
    g_one_star = {}
    for i in gStart:
        g_one_star[i] = {}
    sample_g_single_edge(gStart, g_one_star)
    return all_acceptable
