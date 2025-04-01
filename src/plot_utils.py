import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def top_k_argmax_2d(array, k=5):
    """
    Returns the indices of the k largest values in a 2D numpy array.
    
    Parameters:
        array (numpy.ndarray): 2D input array
        k (int): Number of largest values to find
        
    Returns:
        list: List of (row, col) index tuples for the k largest values
    """
    # Flatten the array to 1D
    non_zero_indices = np.nonzero(array)
    # Get values at those indices
    non_zero_values = array[non_zero_indices]
    
    # If there are no non-zero values, return empty list
    if len(non_zero_values) == 0:
        return []
    
    # Sort the non-zero values and get indices of top k
    k = min(k, len(non_zero_values))  # Ensure k isn't larger than available values
    top_k_indices = np.argsort(non_zero_values)[-k:][::-1]
    
    # Get corresponding row, col indices
    rows = non_zero_indices[0][top_k_indices]
    cols = non_zero_indices[1][top_k_indices]

    return np.array(list(zip(rows, cols)))

def plot_state_visit_and_Q_distr(states_visit, agents, block_start, block_size, action_size, output_dir, top_k=5):
    fig, axes = plt.subplots(figsize=(8, 8))
    plt.title(f'State visit count Over Round {block_start} to {block_start + block_size}', fontsize=16)
    
    states_visit_first_axis_inverted = states_visit[::-1]
    ax = sns.heatmap(data=states_visit_first_axis_inverted, yticklabels=np.arange(len(states_visit))[::-1], xticklabels=np.arange(len(states_visit)), cmap='Blues', linecolor='black', linewidths=0.1)
    
    ax.set(ylabel=agents[0].name, xlabel=agents[1].name)
    if not os.path.exists(f'{output_dir}/state_count'):
        os.makedirs(f'{output_dir}/state_count')
        os.makedirs(f'{output_dir}/Qvalue')
        os.makedirs(f'{output_dir}/ArgmaxStates')
    
    plt.savefig(f'{output_dir}/state_count/{block_start}.pdf', bbox_inches='tight')

    states_to_plot_Q = top_k_argmax_2d(states_visit, top_k)

    all_actions = np.expand_dims(np.arange(action_size), 1)
   
   
    for state in states_to_plot_Q:
        for agent_id, agent in enumerate(agents):
            state_original = state
            if (agent_id == 1):
                state = state[::-1]
            Q_values = agent.bidder.q_values(state, all_actions)
            fig, axes = plt.subplots(figsize=(8, 8))
            state_name = "-".join([str(x) for x in state])
            plt.title(f'The Q-values of State {state_name} (with {states_visit[state_original[0], state_original[1]] } visit) for agent {agent_id} at {block_start + block_size}', fontsize=12)
            plt.scatter(all_actions, Q_values)
            plt.xlabel("Actions")
            plt.ylabel("Q-values")
            plt.savefig(f'{output_dir}/Qvalue/@{block_start}_state_{state_original}_agent_{agent_id}.pdf', bbox_inches='tight')

            
    state_conditional_argmaxes = [agent.get_conditional_argmax() for agent in agents]
    for agent_id, argmaxes in enumerate(state_conditional_argmaxes):
        unique_elements, counts = np.unique(argmaxes, return_counts=True)
        fig, axes = plt.subplots(figsize=(8, 8))
        plt.title(f'The count of states that share the same argmax of Q-values', fontsize=12)
        plt.scatter(unique_elements, counts)
        plt.xlabel("Actions")
        plt.ylabel("State Count")
        plt.savefig(f'{output_dir}/ArgmaxStates/@{block_start}_agent_{agent_id}.pdf', bbox_inches='tight')



def deprecated_plot_state_visit_and_Q_distr(run2agent2measure, measure_name, filename, fontsize):
    # Generate DataFrame for Seaborn
    if type(run2agent2measure) != pd.DataFrame:
        df = measure_per_agent2df(run2agent2measure, measure_name)
    else:
        df = run2agent2measure
    
    # df = df[(df['Iteration'].max() - df['Iteration']) < plot_last_iter]
    max_measure = int(df[measure_name].max()+1)
    block_size = 500
    step_size = 5000
    agents_name = df['Agent'].unique()
    for block_start in range(0, df['Iteration'].max(), step_size):
        block_end = min(df['Iteration'].max() + 1, block_start + block_size)
        df_block = df[df['Iteration'] >= block_start and df['Iteration'] < block_end]
        state_visit_count = np.zeros((max_measure,max_measure))
        fig, axes = plt.subplots(figsize=(8, 8))
        plt.title(f'{measure_name} Over Round {block_start} to {block_end}', fontsize=fontsize + 2)
        for i in range(block_start, block_end):
            df_i = df_block[df_block['Iteration'] == i]
            state_visit_count[-int(df_i.iloc[0][measure_name])-1, int(df_i.iloc[1][measure_name])] += 1

        
        ax = sns.heatmap(data=state_visit_count, yticklabels=np.arange(max_measure)[::-1], xticklabels=np.arange(max_measure), cmap='Blues', linecolor='black', linewidths=0.1)
        
        ax.set(ylabel=agents_name[0], xlabel=agents_name[1])
        plt.savefig(filename, bbox_inches='tight')


def measure_per_agent2df(run2agent2measure, measure_name):
    df_rows = {'Run': [], 'Agent': [], 'Iteration': [], measure_name: []}
    for run, agent2measure in run2agent2measure.items():
        for agent, measures in agent2measure.items():
            for iteration, measure in enumerate(measures):
                df_rows['Run'].append(run)
                df_rows['Agent'].append(agent)
                df_rows['Iteration'].append(iteration)
                df_rows[measure_name].append(measure)
    return pd.DataFrame(df_rows)