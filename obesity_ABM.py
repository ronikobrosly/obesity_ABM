import enum
import itertools
import math
from os.path import expanduser
import pdb
import random
import time

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.isomorphism as iso
from networkx.algorithms.isomorphism import is_isomorphic
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, make_interp_spline
from scipy.signal import savgol_filter
from tqdm import tqdm




random.seed(9901)
np.random.seed(9901)


class GENDER(enum.IntEnum):
    MALE = 0
    FEMALE = 1

class ETHNICITY(enum.IntEnum):
    WHITE = 0
    HISPANIC = 1
    BLACK = 2
    ASIAN = 3

class EDUCATION(enum.IntEnum):
    COLLEGE = 0
    HIGH_SCHOOL = 1
    LT_HIGH_SCHOOL = 2

class INCOME(enum.IntEnum):
    HIGH = 0
    LOW = 1

class NEIGHBORHOOD(enum.IntEnum):
    UPPER_MAJORITY = 0
    UPPER_MINORITY = 1
    LOWER_MAJORITY = 2
    LOWER_MINORITY = 3

class OBESE(enum.IntEnum):
    NO = 0
    YES = 1

STARTING_AGE = 18



def NHANES_pred(age, male, ethnicity, edu, low_income):
    """
    Returns probability of obesity given an individuals demographics
    """
    if ethnicity == ETHNICITY.WHITE:
        hispanic = 0; black = 0; asian = 0;
    elif ethnicity == ETHNICITY.HISPANIC:
        hispanic = 1; black = 0; asian = 0; 
    elif ethnicity == ETHNICITY.BLACK:
        hispanic = 0; black = 1; asian = 0; 
    elif ethnicity == ETHNICITY.ASIAN:
        hispanic = 0; black = 0; asian = 1; 

    if edu == EDUCATION.COLLEGE:
        ls_hs = 0; hs = 0;
    if edu == EDUCATION.HIGH_SCHOOL:
        ls_hs = 0; hs = 1;
    if edu == EDUCATION.LT_HIGH_SCHOOL:
        ls_hs = 1; hs = 0;

    if age <= 24:
        p = (1 / (1 + math.exp(-( -1.945199118947942 + 0.453582197623321*male +  0.594301446380051*hispanic + 0.614344010347805*black +  -0.622784458919374*asian + 1.06944337265171*ls_hs + 0.263091076025842*hs + -0.694036472681529*low_income +  -0.67767823874256*male*hispanic + -1.59238722222648*male*black +  -0.273934330610787*male*asian + 0.846330841389841*hispanic*low_income +  0.845482034114522*black*low_income + 0.717965576609183*asian*low_income ))))  
    elif 25 <= age <= 34:
        p = (1 / (1 + math.exp(-( -1.250510112227838 + 0.284513492553897*male +  -0.106884428592779*hispanic + 0.440639490318372*black +  -1.10989108410785*asian + 0.93342498743326*ls_hs + 0.645048730830948*hs + -0.418999919136112*low_income +  0.233184658178101*male*hispanic + -1.03926745828485*male*black +  0.294450513571392*male*asian + 0.795149019639188*hispanic*low_income +  -0.163579587469932*black*low_income + -0.951472883705997*asian*low_income ))))
    elif 35 <= age <= 44:
        p = (1 / (1 + math.exp(-( -0.674941644905153 + 0.0969952917007317*male +  0.483559801324601*hispanic + 0.60591286535956*black +  -0.934835608962707*asian + 0.146468928501282*ls_hs + 0.646923521999472*hs + -0.123129892523741*low_income +  -0.19442851436543*male*hispanic + -0.678670688661983*male*black +  -0.334458166386686*male*asian + -0.06825395038133*hispanic*low_income +  -0.0765757691718646*black*low_income + -0.371966575825664*asian*low_income ))))
    elif 45 <= age <= 54:
        p = (1 / (1 + math.exp(-( -0.102989247082001 + -0.336404350027203*male +  0.10639118927315*hispanic + 0.909668962331198*black +  -1.67606277312526*asian + 0.107257149717273*ls_hs + 0.213417369991959*hs + -0.118376306434112*low_income +  0.204154600468121*male*hispanic + -0.134545505111619*male*black +  0.620093135764366*male*asian + 0.34930807288916*hispanic*low_income +  -0.201480463397565*black*low_income + 0.420232973505079*asian*low_income ))))    
    elif 55 <= age <= 64:
        p = (1 / (1 + math.exp(-( 0.0410298783766218 + -0.000282839245683864*male +  0.15015773541335*hispanic + 0.535578996430555*black +  -1.93211344172358*asian + 0.157921171423239*ls_hs + 0.0392505882298882*hs + -0.0194247525014116*low_income +  -0.180484576402743*male*hispanic + -0.863405888490444*male*black +  0.246259036524443*male*asian + -0.073786518710037*hispanic*low_income +  -0.0974535632246011*black*low_income + 0.287679145584893*asian*low_income ))))
    elif 65 <= age <= 74:
        p = (1 / (1 + math.exp(-( -0.330002027435108 + 0.038532494720161*male +  0.798420646912457*hispanic + 0.721235546788376*black +  -1.89416154590602*asian + -0.514964239754382*ls_hs + 0.410716017842812*hs + -0.143059890281068*low_income +  -0.477828565732246*male*hispanic + -1.00148134398669*male*black +  -1.10601002826727*male*asian + 0.258604580853212*hispanic*low_income +  -0.00988221015165987*black*low_income + 1.53255330071962*asian*low_income ))))
    elif age >= 75:
        p = (1 / (1 + math.exp(-( -0.487427488057634 + -0.0979824903333087*male +  -1.21968212790741*hispanic + -0.181325606437835*black +  -1.63446131446838*asian + 0.254182721270385*ls_hs + 0.142469848588213*hs + -0.415461935654737*low_income +  1.22255446889448*male*hispanic + -0.297954400569254*male*black +  -14.466648441621*male*asian + -0.145674485357176*hispanic*low_income +  0.362741838826701*black*low_income + 0.575607859670045*asian*low_income ))))

    return p
            





class MyAgent(Agent):
    """ An agent in an obesity model."""
    def __init__(self, unique_id, model, seed = 9901):
        super().__init__(unique_id, model)

        ### Set age
        self.age = STARTING_AGE
        
        ### Set gender
        self.gender = np.random.choice(
            [GENDER.FEMALE, GENDER.MALE], 
            p=[0.5,0.5]
        )
        
        ### Set ethnicity
        self.ethnicity = np.random.choice(
            [ETHNICITY.WHITE, ETHNICITY.HISPANIC, ETHNICITY.BLACK, ETHNICITY.ASIAN],
             p=[0.59, 0.21, 0.13, 0.07]
        )
        
        ### Set education
        if self.ethnicity == ETHNICITY.WHITE:
            self.education = np.random.choice(
                [EDUCATION.COLLEGE, EDUCATION.HIGH_SCHOOL, EDUCATION.LT_HIGH_SCHOOL], 
                p=[0.20, 0.34, 0.46]
            )
        elif self.ethnicity == ETHNICITY.HISPANIC:
            self.education = np.random.choice(
                [EDUCATION.COLLEGE, EDUCATION.HIGH_SCHOOL, EDUCATION.LT_HIGH_SCHOOL], 
                p=[0.07, 0.18, 0.75]
            )
        elif self.ethnicity == ETHNICITY.BLACK:
            self.education = np.random.choice(
                [EDUCATION.COLLEGE, EDUCATION.HIGH_SCHOOL, EDUCATION.LT_HIGH_SCHOOL], 
                p=[0.10, 0.21, 0.69]
            )
        elif self.ethnicity == ETHNICITY.ASIAN:
            self.education = np.random.choice(
                [EDUCATION.COLLEGE, EDUCATION.HIGH_SCHOOL, EDUCATION.LT_HIGH_SCHOOL], 
                p=[0.15, 0.24, 0.61]
            )
        
        ### Set income
        if self.ethnicity == ETHNICITY.WHITE and self.education == EDUCATION.COLLEGE:
            self.income = np.random.choice(
                [INCOME.HIGH, INCOME.LOW], 
                p=[0.85, 0.15]
            )
        elif self.ethnicity == ETHNICITY.WHITE and self.education == EDUCATION.HIGH_SCHOOL:
            self.income = np.random.choice(
                [INCOME.HIGH, INCOME.LOW], 
                p=[0.41, 0.59]
            )
        elif self.ethnicity == ETHNICITY.WHITE and self.education == EDUCATION.LT_HIGH_SCHOOL:
            self.income = np.random.choice(
                [INCOME.HIGH, INCOME.LOW], 
                p=[0.22, 0.78]
            )

        elif self.ethnicity == ETHNICITY.HISPANIC and self.education == EDUCATION.COLLEGE:
            self.income = np.random.choice(
                [INCOME.HIGH, INCOME.LOW], 
                p=[0.78, 0.22]
            )
        elif self.ethnicity == ETHNICITY.HISPANIC and self.education == EDUCATION.HIGH_SCHOOL:
            self.income = np.random.choice(
                [INCOME.HIGH, INCOME.LOW], 
                p=[0.45, 0.55]
            )
        elif self.ethnicity == ETHNICITY.HISPANIC and self.education == EDUCATION.LT_HIGH_SCHOOL:
            self.income = np.random.choice(
                [INCOME.HIGH, INCOME.LOW], 
                p=[0.25, 0.75]
            )

        elif self.ethnicity == ETHNICITY.BLACK and self.education == EDUCATION.COLLEGE:
            self.income = np.random.choice(
                [INCOME.HIGH, INCOME.LOW], 
                p=[0.73, 0.27]
            )
        elif self.ethnicity == ETHNICITY.BLACK and self.education == EDUCATION.HIGH_SCHOOL:
            self.income = np.random.choice(
                [INCOME.HIGH, INCOME.LOW], 
                p=[0.69, 0.31]
            )
        elif self.ethnicity == ETHNICITY.BLACK and self.education == EDUCATION.LT_HIGH_SCHOOL:
            self.income = np.random.choice(
                [INCOME.HIGH, INCOME.LOW], 
                p=[0.18, 0.82]
            )

        elif self.ethnicity == ETHNICITY.ASIAN and self.education == EDUCATION.COLLEGE:
            self.income = np.random.choice(
                [INCOME.HIGH, INCOME.LOW], 
                p=[0.79, 0.21]
            )
        elif self.ethnicity == ETHNICITY.ASIAN and self.education == EDUCATION.HIGH_SCHOOL:
            self.income = np.random.choice(
                [INCOME.HIGH, INCOME.LOW], 
                p=[0.70, 0.30]
            )
        elif self.ethnicity == ETHNICITY.ASIAN and self.education == EDUCATION.LT_HIGH_SCHOOL:
            self.income = np.random.choice(
                [INCOME.HIGH, INCOME.LOW], 
                p=[0.12, 0.88]
            )

        ### Set which neighborhood agent belongs to (of 4 types: based on high or low social class, and minority or majority ethnicity)
        # First, will their social class or ethnicity determine neighborhood?
        social_deter = np.random.choice([1, 0], p=[0.75, 0.25])
        ethnicity_deter = np.random.choice([1, 0], p=[0.75, 0.25])

        # If the social and ethnic status are determined...
        if social_deter == 1 and ethnicity_deter == 1:
            if self.ethnicity == ETHNICITY.WHITE and self.income == INCOME.HIGH:
                self.neighborhood = NEIGHBORHOOD.UPPER_MAJORITY
            if self.ethnicity == ETHNICITY.WHITE and self.income == INCOME.LOW:
                self.neighborhood = NEIGHBORHOOD.LOWER_MAJORITY
            if self.ethnicity != ETHNICITY.WHITE and self.income == INCOME.HIGH:
                self.neighborhood = NEIGHBORHOOD.UPPER_MINORITY
            if self.ethnicity != ETHNICITY.WHITE and self.income == INCOME.LOW:
                self.neighborhood = NEIGHBORHOOD.LOWER_MINORITY

        # If the only social class is determined...
        elif social_deter == 1 and ethnicity_deter == 0:
            
            eth = np.random.choice([1, 0], p=[0.5, 0.5])

            if eth == 1 and self.income == INCOME.HIGH:
                self.neighborhood = NEIGHBORHOOD.UPPER_MAJORITY
            if eth == 1 and self.income == INCOME.LOW:
                self.neighborhood = NEIGHBORHOOD.LOWER_MAJORITY
            if eth == 0 and self.income == INCOME.HIGH:
                self.neighborhood = NEIGHBORHOOD.UPPER_MINORITY
            if eth == 0 and self.income == INCOME.LOW:
                self.neighborhood = NEIGHBORHOOD.LOWER_MINORITY
        
        # If the only ethnicity is determined...
        elif social_deter == 0 and ethnicity_deter == 1:
            
            soc = np.random.choice([1, 0], p=[0.5, 0.5])

            if self.ethnicity == ETHNICITY.WHITE and soc == 1:
                self.neighborhood = NEIGHBORHOOD.UPPER_MAJORITY
            if self.ethnicity == ETHNICITY.WHITE and soc == 0:
                self.neighborhood = NEIGHBORHOOD.LOWER_MAJORITY
            if self.ethnicity != ETHNICITY.WHITE and soc == 1:
                self.neighborhood = NEIGHBORHOOD.UPPER_MINORITY
            if self.ethnicity != ETHNICITY.WHITE and soc == 0:
                self.neighborhood = NEIGHBORHOOD.LOWER_MINORITY

        # If nothing is determined, pick neighborhood randomly...
        elif social_deter == 0 and ethnicity_deter == 0:
            self.neighborhood = np.random.choice(
                [
                    NEIGHBORHOOD.UPPER_MAJORITY,
                    NEIGHBORHOOD.UPPER_MINORITY,
                    NEIGHBORHOOD.LOWER_MAJORITY,
                    NEIGHBORHOOD.LOWER_MINORITY,
                ], 
                p=[0.25, 0.25, 0.25, 0.25]
            )

        ### Set baseline obesity status
        baseline_prob = NHANES_pred(
            age = self.age, 
            male = self.gender,
            ethnicity = self.ethnicity,
            edu = self.education,
            low_income = self.income
        )

        self.obesity_prob = baseline_prob

        self.obesity_status = np.random.choice([OBESE.YES, OBESE.NO], p=[baseline_prob, 1 - baseline_prob])

    def update(self):
        """Check obesity status"""  

            # Add a year to the agent's life           
        self.age += 1

        ### Update status here
        
        # First, update with purely individual-level demographics (only age is updated...)
        update_prob = NHANES_pred(
            age = self.age, 
            male = self.gender,
            ethnicity = self.ethnicity,
            edu = self.education,
            low_income = self.income
        )

        # Apply a neighborhood correction here
        if self.neighborhood == NEIGHBORHOOD.UPPER_MAJORITY:
            update_prob *= 0.91
        elif self.neighborhood == NEIGHBORHOOD.UPPER_MINORITY:
            update_prob *= 0.97
        elif self.neighborhood == NEIGHBORHOOD.LOWER_MAJORITY:
            update_prob *= 1.1
        elif self.neighborhood == NEIGHBORHOOD.LOWER_MINORITY:
            update_prob *= 1.12

        # Determine if anyone connected to agent in network is obese, update prob accordingly
        G_1_neighbors = self.model.G.neighbors(self.graph_1_id)
        G_2_neighbors = self.model.G_2.neighbors(self.graph_1_id)

        obese_neighbors_1 = [
            agent for agent in G_1_neighbors if self.model.G.nodes[agent]['obesity_status'] == OBESE.YES
        ]

        obese_neighbors_2 = [
            agent for agent in G_2_neighbors if self.model.G.nodes[agent]['obesity_status'] == OBESE.YES
        ]

        if obese_neighbors_1:
            update_prob *= 1.07

        if obese_neighbors_2:
            update_prob *= 1.07

        self.obesity_prob = update_prob

        # Make final determine about obesity in this particular time step
        if update_prob >= 1.0:
            update_prob = 0.95

        self.obesity_status = np.random.choice([OBESE.YES, OBESE.NO], p=[update_prob, 1 - update_prob])

        # Update the node in the two graphs to reflect new obesity status
        nx.set_node_attributes(self.model.G, {self.graph_1_id:{'obesity_status': self.obesity_status}})
        nx.set_node_attributes(self.model.G_2, {self.graph_1_id:{'obesity_status': self.obesity_status}})

    def step(self):
        self.update()





class NetworkInfectionModel(Model):
    """A model for infection spread."""
    
    def __init__(self, N=1000, graph_m = 2, seed = 9901):
        
        self.num_nodes = N  
        
        self.schedule = RandomActivation(self)
        self.running = True

        self.G = nx.empty_graph()

        self.grid1 = NetworkGrid(self.G)
        
        ### Create agents and network #1 (Based on education and social class)
        agent_list = []

        node_count = tqdm([i for i in range(self.num_nodes)])
        for i in node_count:

            node_count.set_description("Creating education and social class-based social network")

            # Add the first 2 nodes (0, 1) without any edges
            if i < 2:
                ag = MyAgent(i, self)
                ag.graph_1_id = i
                self.G.add_node(i, ethnicity = ag.ethnicity, education = ag.education, income = ag.income, obesity_status = ag.obesity_status)
                agent_list.append(ag)
            
            # After adding the third node, force connections between nodes 1 and 2 to 0
            elif i == 2:
                ag = MyAgent(i, self)
                ag.graph_1_id = i
                self.G.add_node(i, ethnicity = ag.ethnicity, education = ag.education, income = ag.income, obesity_status = ag.obesity_status)
                agent_list.append(ag)
                self.G.add_edge(0,1)
                self.G.add_edge(0,2)

            # For the first 15% of the new nodes, build a standard Barabási–Albert network, without regard to node attributes
            elif 2 < i < (self.num_nodes * 0.15):
                ag = MyAgent(i, self)
                ag.graph_1_id = i
                self.G.add_node(i, ethnicity = ag.ethnicity, education = ag.education, income = ag.income, obesity_status = ag.obesity_status)
                agent_list.append(ag)
                
                curr_node_list = []
                for key, value in dict(self.G.degree()).items():
                    curr_node_list = curr_node_list + [key]*value

                new_connections = random.choices(curr_node_list, k = graph_m)

                for j in new_connections:
                    self.G.add_edge(i,j)

            # For the middle 50% of the new nodes, restricted to connecting with existing agents of similar ethnicity, 
            # with a probability of connecting to existing agents with the same ethnicity that is proportional 
            # to the number of connections that that agent already possessed
            elif (self.num_nodes * 0.15) <= i < (self.num_nodes * 0.75):
                ag = MyAgent(i, self)
                ag.graph_1_id = i
                agent_list.append(ag)
              
                # Get list of similar nodes
                similar_eth_node_list = [n for n,d in self.G.nodes().items() if d['ethnicity'] == ag.ethnicity]
                
                # Get list of these nodes by their number of connections
                curr_node_list = []
                for node in similar_eth_node_list:
                    curr_node_list = curr_node_list + ([node]*self.G.degree(node))

                new_connections = random.choices(curr_node_list, k = graph_m)

                self.G.add_node(i, ethnicity = ag.ethnicity, education = ag.education, income = ag.income, obesity_status = ag.obesity_status)
                for j in new_connections:
                    self.G.add_edge(i,j)

            # For the last 25% of the new nodes, restricted to connecting with existing agents of similar ethnicity and social class 
            # with a probability of connecting to existing agents with the same ethnicity and social class that is proportional 
            # to the number of connections that that agent already possessed
            elif i >= (self.num_nodes * 0.75):
                ag = MyAgent(i, self)
                ag.graph_1_id = i
                agent_list.append(ag)

                # Get list of similar nodes
                similar_eth_soc_node_list = [n for n,d in self.G.nodes().items() if ((d['ethnicity'] == ag.ethnicity) & (d['income'] == ag.income))]

                # Get list of these nodes by their number of connections
                curr_node_list = []
                for node in similar_eth_soc_node_list:
                    curr_node_list = curr_node_list + ([node]*self.G.degree(node))

                new_connections = random.choices(curr_node_list, k = graph_m)

                self.G.add_node(i, ethnicity = ag.ethnicity, education = ag.education, income = ag.income, obesity_status = ag.obesity_status)
                for j in new_connections:
                    self.G.add_edge(i,j)

        ### Network #2 (Based on neighborhood)
        self.G_2 = nx.empty_graph()

        self.grid2 = NetworkGrid(self.G_2)

        for i, ag in enumerate(agent_list):
            self.G_2.add_node(i, neighborhood = ag.neighborhood)

        # Enumerate all possible node combos
        node_combos = tqdm([i for i in itertools.combinations(self.G_2.nodes(), 2)])
    
        for combo in node_combos:
            node_combos.set_description("Creating neighborhood-based social network")
            tmp_agent_1, tmp_agent_2 = combo
            
            if agent_list[tmp_agent_1].neighborhood == agent_list[tmp_agent_2].neighborhood:
                result = np.random.choice([True, False], p=[(50/self.num_nodes), (1-(50/self.num_nodes))])

            else:
                result = np.random.choice([True, False], p=[(1/self.num_nodes), (1-(1/self.num_nodes))])

            if result:
                self.G_2.add_edge(tmp_agent_1, tmp_agent_2)

        ### Add each agent to the schedule of the simulation
        for i, ag in enumerate(agent_list):
            self.schedule.add(ag)

        ### Let's collect several things from the agent and model classes
        self.datacollector = DataCollector(
            agent_reporters={
                "Obesity_status": "obesity_status",
                "Obesity_prob": "obesity_prob",
                "Age": "age",
                "Ethnicity": "ethnicity",
                "Education": "education",
                "Income": "income",
            },
            model_reporters={
                "Graph_1": "G", 
                "Graph_2": "G_2"
            }
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

        





########### Let's run it! ###########

# How many agents to add?
N=500
# How many years to cover
steps=62
# For some reason I can't collect updated network graphs, so doing manual approach instead of using DataCollector
g1_list = []


model = NetworkInfectionModel(N, graph_m = 2, seed = 9901)
for i in range(steps):
    model.step()

agent_data = model.datacollector.get_agent_vars_dataframe().reset_index()
model_data = model.datacollector.model_vars










#######################################################################################
#######################################################################################
####### ANALYSIS #######


### Let's ensure that the networks are actually updating

# obese_agent_list_1 = [agent for agent in g1_list[0].nodes if g1_list[0].nodes[agent]['obesity_status'] == OBESE.YES]
# obese_agent_list_2 = [agent for agent in g1_list[5].nodes if g1_list[10].nodes[agent]['obesity_status'] == OBESE.YES]

# print(obese_agent_list_1 == obese_agent_list_2)

# obe_match = iso.numerical_node_match("obesity_status", 0)
# print(is_isomorphic(g1_list[0], g1_list[5], node_match = obe_match))








### Plot probability of obesity for an individual agent over time
agent_id = 18

age = np.arange(start = 18, stop = 80, step = 1)
prob = agent_data[agent_data['AgentID'] == agent_id]['Obesity_prob']

new_y = savgol_filter(prob, 7, 3)

plt.figure(figsize=(12, 8))
ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
plt.plot(age, new_y)
plt.ylim(0, 1.0)
plt.title(f'Probability of Obesity for Agent {agent_id}')
plt.xlabel('Age in birth cohort')
plt.ylabel('Probability of obesity')
plt.savefig(expanduser("~/Desktop/single_agent_prob.png"), dpi = 300)








### Plot prevalence of obesity in this birth-cohort over time
age = np.arange(start = 18, stop = 80, step = 1)
prev = agent_data.groupby('Step')['Obesity_status'].mean().values

new_y = savgol_filter(prev, 21, 3)

plt.figure(figsize=(12, 8))
ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
plt.plot(age, new_y)
plt.ylim(0, 1.0)
plt.title('')
plt.xlabel('Age in birth cohort')
plt.ylabel('Prevalence of obesity')
plt.savefig(expanduser("~/Desktop/prevalence_by_age.png"), dpi = 300)





### Plot prevalence of obesity by ethnicity
plt.figure(figsize=(12, 8))
ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  

age = np.arange(start = 18, stop = 80, step = 1)

tmp_data = agent_data[agent_data['Ethnicity'] == ETHNICITY.WHITE]
prev = tmp_data.groupby('Step')['Obesity_status'].mean().values
new_y = savgol_filter(prev, 21, 3)
plt.plot(age, new_y, color = '#ff7f0e')

tmp_data = agent_data[agent_data['Ethnicity'] == ETHNICITY.HISPANIC]
prev = tmp_data.groupby('Step')['Obesity_status'].mean().values
new_y = savgol_filter(prev, 21, 3)
plt.plot(age, new_y, color = '#1f77b4')

tmp_data = agent_data[agent_data['Ethnicity'] == ETHNICITY.BLACK]
prev = tmp_data.groupby('Step')['Obesity_status'].mean().values
new_y = savgol_filter(prev, 21, 3)
plt.plot(age, new_y, color = '#2ca02c')

tmp_data = agent_data[agent_data['Ethnicity'] == ETHNICITY.ASIAN]
prev = tmp_data.groupby('Step')['Obesity_status'].mean().values
new_y = savgol_filter(prev, 21, 3)
plt.plot(age, new_y, color = '#e377c2')

plt.ylim(0, 1.0)
plt.title('')
plt.xlabel('Age in birth cohort')
plt.ylabel('Prevalence of obesity')

legend_elements = [
    Patch(facecolor='#ff7f0e', label='White'),
    Patch(facecolor='#1f77b4', label='Hispanic'),
    Patch(facecolor='#2ca02c', label='Black'),
    Patch(facecolor='#e377c2', label='Asian')
]
plt.legend(handles=legend_elements)
plt.savefig(expanduser("~/Desktop/prevalence_by_age_by_race.png"), dpi = 300)










### Plot education and social class network cluster (G)
test_G = model_data['Graph_1'][0]
pos = nx.spring_layout(test_G, scale=20)
color_state_map = {0: '#ff7f0e', 1: '#1f77b4', 2: '#2ca02c', 3: '#e377c2'}

plt.figure(figsize=(12, 8))
nx.draw(
    test_G, 
    pos=pos, 
    with_labels=False, 
    node_color=[color_state_map[node[1]['ethnicity']] for node in test_G.nodes(data=True)], 
    edge_color="dimgrey",
    width=0.5,
    node_size=65,
    font_color='white'
)
legend_elements = [
    Patch(facecolor='#ff7f0e', label='White'),
    Patch(facecolor='#1f77b4', label='Hispanic'),
    Patch(facecolor='#2ca02c', label='Black'),
    Patch(facecolor='#e377c2', label='Asian')
]
plt.legend(handles=legend_elements)
plt.savefig(expanduser("~/Desktop/edu_class_network.png"), dpi = 300)





### Plot the 4 neighorhood network clusters (G_2)
test_G = model_data['Graph_2'][0]
pos = nx.spring_layout(test_G, scale=20)
color_state_map = {0: '#ff7f0e', 1: '#1f77b4', 2: '#2ca02c', 3: '#e377c2'}

plt.figure(figsize=(12, 8))
nx.draw(
    test_G, 
    pos=pos, 
    with_labels=False, 
    node_color=[color_state_map[node[1]['neighborhood']] for node in test_G.nodes(data=True)], 
    edge_color="dimgrey",
    width=0.5,
    node_size=65,
    font_color='white'
)
legend_elements = [
    Patch(facecolor='#ff7f0e', label='Upper Class, White Majority'),
    Patch(facecolor='#1f77b4', label='Upper Class, Non-White Majority'),
    Patch(facecolor='#2ca02c', label='Lower Class, White Majority'),
    Patch(facecolor='#e377c2', label='Lower Class, Non-White Majority')
]
plt.legend(handles=legend_elements)
plt.savefig(expanduser("~/Desktop/neighborhood_network.png"), dpi = 300)







