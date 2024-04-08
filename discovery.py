from causallearn.search.ConstraintBased.PC import pc


#
# import unittest
#
#
#
#
# #
# # chang shi GPT!!!!!!c
# #
# class TestPC(unittest.TestCase):
#
#   def test(self):
#      data=''
#
#
#      #delayVar Select
#      delaySetp = 5
#
#      disResults=[]
#      score=0
#      for i in  range(10):
#        if score < 90:
#          data = TestPC.comstructData(delaySetp)
#          cg = pc(data)
#          ResultsG = cg.G
#          #todo
#          #serach deay var in G
#          #todo
#          #change delaySetp
#          delaySetp= 0
#
#
#   def comstructData(self,delaySetp):
#
#     data = 1
#     delaySetp = 1
#     return data



import networkx as nx  # Import the NetworkX library

import numpy as np
from causallearn.search.ConstraintBased.PC import pc

def lag_variable(data, lag):
    return np.roll(data, lag)


def calculate_bic(data, graph):
    num_samples, num_variables = data.shape
    num_edges = len(graph.get_graph_edges())
    log_likelihood = 0

    for variable in range(num_variables):
        parents = list(graph.predecessors(variable))
        conditional_data = np.column_stack([data[:, parent] for parent in parents] + [data[:, variable]])
        mean = np.mean(conditional_data, axis=0)
        cov = np.cov(conditional_data, rowvar=False)
        log_likelihood += np.sum(np.log(np.diag(np.linalg.cholesky(cov)))) - 0.5 * num_samples * (
                    num_variables + 1) * np.log(2 * np.pi) - 0.5 * np.sum(
            (data[:, variable] - mean) @ np.linalg.inv(cov) @ (data[:, variable] - mean))

    bic = log_likelihood - 0.5 * num_edges * np.log(num_samples)  # BIC formula
    return bic


def main(data, max_lag, max_iterations, bic_threshold):
    num_samples, num_variables = data.shape
    lagged_data = [data]

    for lag in range(1, max_lag + 1):
        lagged = np.apply_along_axis(lambda col: lag_variable(col, lag), axis=0, arr=data)
        lagged_data.append(lagged)

    prev_bic = float('-inf')
    for iteration in range(max_iterations):
        all_edges = set()
        for lagged_data_step in lagged_data:
            rs = pc(data=lagged_data_step)
            graph= rs.G
            # graph = rs.fit(return_type='graph')

            all_edges.update((edge.node1,edge.node2) for edge in graph.get_graph_edges()) # Using tuples

        # Result Analysis
        effective_edges = set()
        for edge in all_edges:
            cause, effect = edge
            if any((cause, effect) in edge_set or (effect, cause) in edge_set for edge_set in
                   [all_edges, effective_edges]):
                continue
            effective_edges.add(edge)
        #update lagged_data
        updated_lagged_data = [data]
        for lag in range(1, max_lag + 1):
            lagged = np.apply_along_axis(lambda col: lag_variable(col, lag), axis=0, arr=data)
            for cause, effect in effective_edges:
                if cause in lagged_data[lag - 1] and effect in lagged_data[0]:
                    updated_lagged_data.append(lagged)
                    break
        lagged_data = updated_lagged_data

        final_model_edges = set()
        for lagged_data_step in lagged_data:
            rs = pc(data=lagged_data_step)
            graph= rs.G.graph
            # graph = rs.fit(return_type='graph')
            final_model_edges.update(edge.__str__() for edge in graph.get_graph_edges())   # Using tuples

        current_bic = calculate_bic(data, graph)

        if iteration > 0 and abs(current_bic - prev_bic) < bic_threshold:
            break

        prev_bic = current_bic

    return final_model_edges


# Generate some example time series data (replace this with your own data)
np.random.seed(0)
num_samples = 100
num_variables = 4
data = np.random.randn(num_samples, num_variables)

max_lag = 5
max_iterations = 10
bic_threshold = 10

final_edges = main(data, max_lag, max_iterations, bic_threshold)
print("Final Causal Relationships:")
for edge in final_edges:
    print(edge)





