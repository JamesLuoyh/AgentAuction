import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

from Agent import Agent
from AuctionAllocation import * # FirstPrice, SecondPrice
from Auction import Auction
from Bidder import *  # EmpiricalShadedBidder, TruthfulBidder
from DQNBidder import *
from BidderAllocation import *  #  LogisticTSAllocator, OracleAllocator

import pickle
from plot_utils import *

def parse_kwargs(kwargs):
    parsed = ','.join([f'{key}={value}' for key, value in kwargs.items()])
    return ',' + parsed if parsed else ''


def parse_config(path):
    with open(path) as f:
        config = json.load(f)

    # Set up Random Number Generator
    rng = np.random.default_rng(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Number of runs
    num_runs = config['num_runs'] if 'num_runs' in config.keys() else 1
    
    # Max. number of slots in every auction round
    # Multi-slot is currently not fully supported.
    max_slots = 1

    # Technical parameters for distribution of latent embeddings
    embedding_size = config['embedding_size']
    plot_last_iter = config.get('plot_last_iter',config['num_iter'])
    avg_over = config.get('avg_over',1)
    embedding_var = config['embedding_var']
    obs_embedding_size = config['obs_embedding_size']

    # Expand agent-config if there are multiple copies
    agent_configs = []
    num_agents = 0
    for agent_config in config['agents']:
        if 'num_copies' in agent_config.keys():
            for i in range(1, agent_config['num_copies'] + 1):
                agent_config_copy = deepcopy(agent_config)
                agent_config_copy['name'] += f' {num_agents + 1}'
                agent_configs.append(agent_config_copy)
                num_agents += 1
        else:
            agent_configs.append(agent_config)
            num_agents += 1

    # First sample item catalog (so it is consistent over different configs with the same seed)
    # Agent : (item_embedding, item_value)
    agents2items = {
        agent_config['name']: rng.normal(0.0, embedding_var, size=(agent_config['num_items'], embedding_size))
        for agent_config in agent_configs
    }
    agents2item_values = {
        agent_config['name']: agent_config['value'] * np.ones(agent_config['num_items']) #rng.lognormal(0.1, 0.2, agent_config['num_items'])
        for agent_config in agent_configs
    }

    # Add intercepts to embeddings (Uniformly in [-4.5, -1.5], this gives nicer distributions for P(click))
    for agent, items in agents2items.items():
        agents2items[agent] = np.hstack((items, - 3.0 - 1.0 * rng.random((items.shape[0], 1))))

    return rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size, plot_last_iter, avg_over


def instantiate_agents(rng, agent_configs, agents2item_values, agents2items):
    # Store agents to be re-instantiated in subsequent runs
    # Set up agents
    
    agents = []

    for agent_config in agent_configs:
        if bool(agent_config.get("pretrained_path")):
            filehandler = open(agent_config.get("pretrained_path"), 'rb') 
            agent = pickle.load(filehandler)
            agent.name = agent_config['name']
            agent.bidder.override_params(agent_config.get('bidder_params_override', {}))
            agents.append(agent)
        else:
            agents.append(Agent(rng=rng,
                name=agent_config['name'],
                num_items=agent_config['num_items'],
                item_values=agents2item_values[agent_config['name']],
                allocator=eval(f"{agent_config['allocator']['type']}(rng=rng{parse_kwargs(agent_config['allocator']['kwargs'])})"),
                bidder=eval(f"{agent_config['bidder']['type']}(rng=rng{parse_kwargs(agent_config['bidder']['kwargs'])})"),
                memory=(0 if 'memory' not in agent_config.keys() else agent_config['memory'])))

    for agent in agents:
        if isinstance(agent.allocator, OracleAllocator) or isinstance(agent.allocator, FixedAllocator):
            agent.allocator.update_item_embeddings(agents2items[agent.name])

    return agents


def instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots, embedding_size, embedding_var, obs_embedding_size):
    return (Auction(rng,
                    eval(f"{config['allocation']}()"),
                    agents,
                    agents2items,
                    agents2item_values,
                    max_slots,
                    embedding_size,
                    embedding_var,
                    obs_embedding_size,
                    config['num_participants_per_round']),
            config['num_iter'], config['rounds_per_iter'], config['output_dir'])


def simulation_run():
    stable = 0
    step_size = 5000
    block_size = 500
    ignore_first = 0#10000
    count = 0
    start_recording = False
    for i in range(num_iter):
 
        for _ in range(rounds_per_iter):
            auction.simulate_opportunity()

        names = [agent.name for agent in auction.agents]
        net_utilities = [agent.net_utility for agent in auction.agents]
        gross_utilities = [agent.gross_utility for agent in auction.agents]
        action_size = auction.agents[0].action_size()
        result = pd.DataFrame({'Name': names, 'Net': net_utilities, 'Gross': gross_utilities})

        if i%10000 == 0:
            print(f'==== ITERATION {i} ====')
           
            print(result)
            print(f'\tAuction revenue: \t {auction.revenue}')

        # early_stopping = True
        if i%step_size==0 and i >= ignore_first:
            start_recording = True
            count = 0
            states_visit = np.zeros([action_size, action_size])
        if count == block_size or count == num_iter - 1:
            # generate plot of the Q values corresponding to the top 5 visited states in the past block_size round
            plot_state_visit_and_Q_distr(states_visit, agents, i-count, count, action_size, output_dir)
            start_recording = False
            count = 0
        if start_recording:
            count += 1
            # Assuming 2 agents for now
            states_visit[int(agents[0].total_bid), int(agents[1].total_bid)] += 1

        for agent_id, agent in enumerate(auction.agents):
            agent.update(iteration=i, plot=True, figsize=FIGSIZE, fontsize=FONTSIZE)

            agent2net_utility[agent.name].append(agent.net_utility)
            agent2gross_utility[agent.name].append(agent.gross_utility)

            agent2allocation_regret[agent.name].append(agent.get_allocation_regret())
            agent2estimation_regret[agent.name].append(agent.get_estimation_regret())
            agent2overbid_regret[agent.name].append(agent.get_overbid_regret())
            agent2underbid_regret[agent.name].append(agent.get_underbid_regret())


            agent2CTR_RMSE[agent.name].append(agent.get_CTR_RMSE())
            agent2CTR_bias[agent.name].append(agent.get_CTR_bias())
            agent2bid[agent.name].append(agent.total_bid)
            agent2argmax[agent.name].append(agent.argmax)
            if isinstance(agent.bidder, PolicyLearningBidder) or isinstance(agent.bidder, DoublyRobustBidder):
                agent2gamma[agent.name].append(torch.mean(torch.Tensor(agent.bidder.gammas)).detach().item())
            elif not agent.bidder.truthful:
                agent2gamma[agent.name].append(np.mean(agent.bidder.gammas))

            best_expected_value = np.mean([opp.best_expected_value for opp in agent.logs])
            agent2best_expected_value[agent.name].append(best_expected_value)

            agent2conditional_argmax[agent.name].append(agent.get_conditional_argmax())
            

            # print('Average Best Value for Agent: ', best_expected_value)
            agent.clear_utility()
            agent.clear_logs()

            # if len(agent2net_utility[agent.name]) < 2 * stopping_criteria or np.sum(agent2net_utility[agent.name][-stopping_criteria:]) > np.sum(agent2net_utility[agent.name][-2*stopping_criteria:-stopping_criteria]):
            #     early_stopping = False
            argmax_history = agent2conditional_argmax[agent.name]
            same_argmax = np.min(argmax_history[max(len(argmax_history) - 2, 0)] == argmax_history[-1])
            stable = (stable + same_argmax) * same_argmax
            auction_revenue.append(auction.revenue)
        auction.clear_revenue()
        if stable > 200000:
            print("CONVERGED!")
            return
        # if early_stopping:
        #     return

def store_agents(agents, config, agent_configs, output_dir):
    for i, agent in enumerate(agents):
        if agent_configs[i].get('save_policy', False):
            filehandler = open(f'{output_dir}/agent_{i}_seed_{config["random_seed"]}.pkl', 'wb') 
            pickle.dump(agent, filehandler)

if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    # Parse configuration file
    rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size, plot_last_iter, avg_over = parse_config(args.config)
    #  = 1#00
    # Plotting config
    FIGSIZE = (8, 5)
    FONTSIZE = 14

    # Placeholders for summary statistics over all runs
    run2agent2net_utility = {}
    run2agent2gross_utility = {}
    run2agent2allocation_regret = {}
    run2agent2estimation_regret = {}
    run2agent2overbid_regret = {}
    run2agent2underbid_regret = {}
    run2agent2best_expected_value = {}

    run2agent2CTR_RMSE = {}
    run2agent2CTR_bias = {}
    run2agent2gamma = {}
    run2agent2bid = {}
    run2agent2argmax = {}
    
    run2agent2conditional_argmax = {}


    run2auction_revenue = {}
    # Make sure we can write results
    
    # Repeated runs
    for run in range(num_runs):
        # Reinstantiate agents and auction per run
        agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
        auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots, embedding_size, embedding_var, obs_embedding_size)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Placeholders for summary statistics per run
        agent2net_utility = defaultdict(list)
        agent2gross_utility = defaultdict(list)
        agent2allocation_regret = defaultdict(list)
        agent2estimation_regret = defaultdict(list)
        agent2overbid_regret = defaultdict(list)
        agent2underbid_regret = defaultdict(list)
        agent2best_expected_value = defaultdict(list)

        agent2CTR_RMSE = defaultdict(list)
        agent2CTR_bias = defaultdict(list)
        agent2gamma = defaultdict(list)
        agent2bid = defaultdict(list)
        agent2argmax = defaultdict(list)

        agent2conditional_argmax = defaultdict(list)

        auction_revenue = []

        # Run simulation (with global parameters -- fine for the purposes of this script)
        simulation_run()

        # Store
        run2agent2net_utility[run] = agent2net_utility
        run2agent2gross_utility[run] = agent2gross_utility
        run2agent2allocation_regret[run] = agent2allocation_regret
        run2agent2estimation_regret[run] = agent2estimation_regret
        run2agent2overbid_regret[run] = agent2overbid_regret
        run2agent2underbid_regret[run] = agent2underbid_regret
        run2agent2best_expected_value[run] = agent2best_expected_value

        run2agent2CTR_RMSE[run] = agent2CTR_RMSE
        run2agent2CTR_bias[run] = agent2CTR_bias
        run2agent2gamma[run] = agent2gamma

        run2auction_revenue[run] = auction_revenue
        run2agent2bid[run] = agent2bid
        run2agent2argmax[run] = agent2argmax
        run2agent2conditional_argmax[run] = agent2conditional_argmax


    store_agents(agents, config, agent_configs, output_dir)



    def heatmap_measure_per_agent(run2agent2measure, measure_name, cumulative=False, log_y=False, yrange=None, optimal=None):
        # Generate DataFrame for Seaborn
        if type(run2agent2measure) != pd.DataFrame:
            df = measure_per_agent2df(run2agent2measure, measure_name)
        else:
            df = run2agent2measure

        fig, axes = plt.subplots(figsize=(8, 8))
        plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
        df = df[(df['Iteration'].max() - df['Iteration']) < plot_last_iter]
        max_measure = int(df[measure_name].max()+1)
        heat = np.zeros((max_measure,max_measure))
        max_itr = df['Iteration'].max()
        for i in range(min(max_itr,plot_last_iter)):
            df_i = df[df['Iteration'] == max_itr - i]
            heat[-int(df_i.iloc[0][measure_name])-1, int(df_i.iloc[1][measure_name])] += 1
        ax = sns.heatmap(data=heat, yticklabels=np.arange(max_measure)[::-1], xticklabels=np.arange(max_measure), cmap='Blues', linecolor='black', linewidths=0.1)
        agents_name = df['Agent'].unique()
        ax.set(ylabel=agents_name[0], xlabel=agents_name[1])
        plt.savefig(f"{output_dir}/heatmap_{measure_name.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.pdf", bbox_inches='tight')




    def heatmap_measure_per_agent_in_blocks(run2agent2measure, measure_name, cumulative=False, log_y=False, yrange=None, optimal=None):
        # Generate DataFrame for Seaborn
        if type(run2agent2measure) != pd.DataFrame:
            df = measure_per_agent2df(run2agent2measure, measure_name)
        else:
            df = run2agent2measure
        
        # df = df[(df['Iteration'].max() - df['Iteration']) < plot_last_iter]
        max_measure = int(df[measure_name].max()+1)
        block_size = 500
        agents_name = df['Agent'].unique()
        # for block in range(df['Iteration'].max() // block_size + 1):
        for block in range(df['Iteration'].max() // block_size, df['Iteration'].max() // block_size + 1):
            df_block = df[df['Iteration'] // block_size == block]
            heat = np.zeros((max_measure,max_measure))
            # max_itr = df_block['Iteration'].max()
            fig, axes = plt.subplots(figsize=(8, 8))
            plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
            for i in range(block_size * block, min(df['Iteration'].max(), block_size * (block+1))):
                df_i = df_block[df_block['Iteration'] == i]
                heat[-int(df_i.iloc[0][measure_name])-1, int(df_i.iloc[1][measure_name])] += 1
            ax = sns.heatmap(data=heat, yticklabels=np.arange(max_measure)[::-1], xticklabels=np.arange(max_measure), cmap='Blues', linecolor='black', linewidths=0.1)
            
            ax.set(ylabel=agents_name[0], xlabel=agents_name[1])
            plt.savefig(f"{output_dir}/heatmap_{block}_{measure_name.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.pdf", bbox_inches='tight')


    def plot_measure_per_agent(run2agent2measure, measure_name, cumulative=False, log_y=False, yrange=None, optimal=None):
        # Generate DataFrame for Seaborn
        if type(run2agent2measure) != pd.DataFrame:
            df = measure_per_agent2df(run2agent2measure, measure_name)
        else:
            df = run2agent2measure

        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
        min_measure, max_measure = 0.0, 0.0
        df = df[(df['Iteration'].max() - df['Iteration']) < plot_last_iter]
        
        df['avg_id'] = np.floor(df['Iteration'] / avg_over)
        df_avg = df.groupby(['Agent', 'Run', 'avg_id']).mean().reset_index()
        sns.lineplot(data=df_avg, x="Iteration", y=measure_name, hue="Agent", ax=axes)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        if optimal is not None:
            plt.axhline(optimal, ls='--', color='gray', label='Optimal')
            min_measure = min(min_measure, optimal)
        if log_y:
            plt.yscale('log')
        if yrange is None:
            factor = 1.1 if min_measure < 0 else 0.9
            # plt.ylim(min_measure * factor, max_measure * 1.1)
        else:
            plt.ylim(yrange[0], yrange[1])
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.legend(loc='upper left', bbox_to_anchor=(-.05, -.15), fontsize=FONTSIZE, ncol=3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.pdf", bbox_inches='tight')
        # plt.show()
        return df, df_avg


    def plot_conditional_argmax_per_agent(run2agent2conditional_argmax, measure_name):
        # Generate DataFrame for Seaborn

        for key in run2agent2conditional_argmax[0]:
            # for i in range(0, len(run2agent2conditional_argmax[0][key]), 5000):
            #     # fig, axes = plt.subplots(figsize=FIGSIZE)
            #     # plt.title(f'{measure_name}@{i} for {key}', fontsize=FONTSIZE + 2)
            #     # plot_axes = plt.axes(projection = '3d')
            #     # plot_axes.set_xlabel('last bid')
            #     # plot_axes.set_ylabel('last opponent bid')
            #     # plot_axes.set_zlabel('argmax')
            #     # conditional_argmax = run2agent2conditional_argmax[0][key][i]
            #     # x, y = np.indices(conditional_argmax.shape)
            #     # x = x.flatten()  # Convert to 1D arrays
            #     # y = y.flatten()
            #     # z = conditional_argmax.flatten()

            #     # # Create the 3D scatter plot
            #     # # fig = plt.figure()

            #     # plot_axes.scatter3D(x, y, z)
    
            #     # # plt.zticks(fontsize=FONTSIZE - 2)
            #     # # plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
            #     # plt.savefig(f"{output_dir}/{f'{measure_name}@{i} for {key}'.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.pdf", bbox_inches='tight')
            #     print(i, key, run2agent2conditional_argmax[0][key][i])
            print(key, run2agent2conditional_argmax[0][key][-1])

    net_utility_df, net_utility_df_avg = plot_measure_per_agent(run2agent2net_utility, 'Net Utility')
    net_utility_df = net_utility_df.sort_values(['Agent', 'Run', 'Iteration'])
    net_utility_df_avg = net_utility_df_avg.sort_values(['Agent', 'Run', 'avg_id'])
    net_utility_df.to_csv(f'{output_dir}/net_utility_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)
    net_utility_df_avg.to_csv(f'{output_dir}/net_utility_avg_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)

    net_utility_df['Net Utility (Cumulative)'] = net_utility_df.groupby(['Agent', 'Run'])['Net Utility'].cumsum()
    plot_measure_per_agent(net_utility_df, 'Net Utility (Cumulative)')

    gross_utility_df, gross_utility_df_avg = plot_measure_per_agent(run2agent2gross_utility, 'Gross Utility')
    gross_utility_df = gross_utility_df.sort_values(['Agent', 'Run', 'Iteration'])
    gross_utility_df.to_csv(f'{output_dir}/gross_utility_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)

    gross_utility_df['Gross Utility (Cumulative)'] = gross_utility_df.groupby(['Agent', 'Run'])['Gross Utility'].cumsum()
    plot_measure_per_agent(gross_utility_df, 'Gross Utility (Cumulative)')

    plot_measure_per_agent(run2agent2best_expected_value, 'Mean Expected Value for Top Ad')

    plot_measure_per_agent(run2agent2allocation_regret, 'Allocation Regret')
    plot_measure_per_agent(run2agent2estimation_regret, 'Estimation Regret')
    overbid_regret_df, overbid_regret_df_avg = plot_measure_per_agent(run2agent2overbid_regret, 'Overbid Regret')
    overbid_regret_df.to_csv(f'{output_dir}/overbid_regret_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)
    underbid_regret_df, underbid_regret_df_avg = plot_measure_per_agent(run2agent2underbid_regret, 'Underbid Regret')
    underbid_regret_df.to_csv(f'{output_dir}/underbid_regret_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)

    plot_measure_per_agent(run2agent2CTR_RMSE, 'CTR RMSE', log_y=True)
    plot_measure_per_agent(run2agent2CTR_bias, 'CTR Bias', optimal=1.0) #, yrange=(.5, 5.0))

    shading_factor_df = plot_measure_per_agent(run2agent2gamma, 'Shading Factors')

    heatmap_measure_per_agent(run2agent2bid, 'Bid')
    bid_df, bid_df_avg = plot_measure_per_agent(run2agent2bid, 'Bid')
    bid_df = bid_df.sort_values(['Agent', 'Run', 'Iteration'])
    bid_df.to_csv(f'{output_dir}/bid_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)

    heatmap_measure_per_agent_in_blocks(run2agent2argmax, 'Q-Max')
    q_df, q_df_avg = plot_measure_per_agent(run2agent2argmax, 'Q-Max')
    q_df = q_df.sort_values(['Agent', 'Run', 'Iteration'])
    q_df.to_csv(f'{output_dir}/qmax_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)

    def measure2df(run2measure, measure_name):
        df_rows = {'Run': [], 'Iteration': [], measure_name: []}
        for run, measures in run2measure.items():
            for iteration, measure in enumerate(measures):
                df_rows['Run'].append(run)
                df_rows['Iteration'].append(iteration)
                df_rows[measure_name].append(measure)
        return pd.DataFrame(df_rows)

    def plot_measure_overall(run2measure, measure_name):
        # Generate DataFrame for Seaborn
        if type(run2measure) != pd.DataFrame:
            df = measure2df(run2measure, measure_name)
        else:
            df = run2measure
        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
        sns.lineplot(data=df, x="Iteration", y=measure_name, ax=axes)
        min_measure = min(0.0, np.min(df[measure_name]))
        max_measure = max(0.0, np.max(df[measure_name]))
        plt.xlabel('Iteration', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        factor = 1.1 if min_measure < 0 else 0.9
        plt.ylim(min_measure * factor, max_measure * 1.1)
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.pdf", bbox_inches='tight')
        # plt.show()
        return df

    auction_revenue_df = plot_measure_overall(run2auction_revenue, 'Auction Revenue')

    net_utility_df_overall = net_utility_df.groupby(['Run', 'Iteration'])['Net Utility'].sum().reset_index().rename(columns={'Net Utility': 'Social Surplus'})
    plot_measure_overall(net_utility_df_overall, 'Social Surplus')

    gross_utility_df_overall = gross_utility_df.groupby(['Run', 'Iteration'])['Gross Utility'].sum().reset_index().rename(columns={'Gross Utility': 'Social Welfare'})
    plot_measure_overall(gross_utility_df_overall, 'Social Welfare')

    plot_conditional_argmax_per_agent(run2agent2conditional_argmax, "Conditional argmax")

    auction_revenue_df['Measure Name'] = 'Auction Revenue'
    net_utility_df_overall['Measure Name'] = 'Social Surplus'
    gross_utility_df_overall['Measure Name'] = 'Social Welfare'

    columns = ['Run', 'Iteration', 'Measure', 'Measure Name']
    auction_revenue_df.columns = columns
    net_utility_df_overall.columns = columns
    gross_utility_df_overall.columns = columns

    pd.concat((auction_revenue_df, net_utility_df_overall, gross_utility_df_overall)).to_csv(f'{output_dir}/results_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)
