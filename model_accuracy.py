import warnings

# statsbombpy likes to remind you that you are accessing a free dataset with every request :)
warnings.filterwarnings('ignore', module='statsbombpy')

# create a dataframe of shots from statsbomb events data
import numpy as np
from statsbombpy import sb
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars

# bring in parameters of homemade xg model
p = {'intercept': 1.34, 'distance': 0.0854, 'angle': -1.1953}


# function for calculating pgoal using DA_ET5L_1516 model
def get_pgoal(x, y):
    x = 120 - x
    c = abs(y - 40)
    d = np.sqrt(x ** 2 + abs(c / 2) ** 2)
    a = np.arctan(7.32 * x / (x ** 2 + abs(c / 2) ** 2 - (7.32 / 2) ** 2))
    pgoal = 1 / (1 + np.exp(p['intercept'] + (p['distance'] * d) + (p['angle'] * a)))
    return pgoal

# calculate xG using custom model and add to a set of events data
test_data = pd.read_csv('test_data.csv')
for i, e in test_data.iterrows():
    test_data.at[i,'type'] = eval(e['shot'])['type']['name']
    x = float(e['location'].strip('][').split(',')[0])
    y = float(e['location'].strip('][').split(',')[1])
    pgoal = get_pgoal(x,y)

    test_data.at[i, 'type'] = eval(e['shot'])['type']['name']
    test_data.at[i, 'statsbomb_xg'] = eval(e['shot'])['statsbomb_xg']
    test_data.at[i, 'da-et5l-1516_xg'] = pgoal


# write xG totals to a match summary table
matches = pd.read_csv('test_data_matches.csv')
for i, m in matches.iterrows():
    e = test_data[test_data['match_id'] == m['match_id']]
    e = e[e['type'] == 'Open Play']
    e_home = e[e['team'] == m['home_team']]
    e_away = e[e['team'] == m['away_team']]
    matches.at[i, 'statsbomb_home_xg'] = e_home['statsbomb_xg'].sum()
    matches.at[i, 'statsbomb_away_xg'] = e_away['statsbomb_xg'].sum()
    matches.at[i, 'daet5l1516_home_xg'] = e_home['da-et5l-1516_xg'].sum()
    matches.at[i, 'daet5l1516_away_xg'] = e_away['da-et5l-1516_xg'].sum()

matches.to_csv('model_analysis.csv')

########################################################################################################

# error analysis
# compare models based on goals predicted vs actual goals scored
# for each match, calculate error as a percent of total (n_goals_predicted - n_goals / n_goals)
# repeat for each team in a given match
# repeat using both models
sb_er = []
my_er = []

for x, m in matches.iterrows():

    # aggregate sb error by team performance
    if m['home_score'] > 0:
        sb_er.append((m['statsbomb_home_xg'] - m['home_score']) / m['home_score'])
    else:
        sb_er.append(m['statsbomb_home_xg'])
    if m['away_score'] > 0:
        sb_er.append((m['statsbomb_away_xg'] - m['away_score']) / m['away_score'])
    else:
        sb_er.append(m['statsbomb_away_xg'])

    # aggregate my error by team performance
    if m['home_score'] > 0:
        my_er.append((m['daet5l1516_home_xg'] - m['home_score']) / m['home_score'])
    else:
        my_er.append(m['daet5l1516_home_xg'])
    if m['away_score'] > 0:
        my_er.append((m['daet5l1516_away_xg'] - m['away_score']) / m['away_score'])
    else:
        my_er.append(m['daet5l1516_away_xg'])

df = pd.DataFrame({'Statsbomb': sb_er, 'DA_ET5L_1516': my_er})

# generate boxplot to visualize accuracy
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot()
ax.boxplot(df, vert=False)
ax.set_yticklabels(['statsbomb', 'DA_ET5L_1516'])
ax.set_title("xG Model Goal Prediction Accuracy: 2022 Men's World Cup")
plt.tight_layout()
plt.savefig('model_accuracy.png')

# print results to terminal
print(df.describe().round(3))



########################################################################################################
fig, ax = plt.subplots()

s = 25
n = 10
ticks = []

for i, e in enumerate(range(n)):
    e = np.random.rand(s)
    plt.plot(e, len(e) * [i+1], "x", color='grey')
    plt.plot(np.mean(e), [i+1], "o", color='red', label='MAE')
    plt.plot(np.sqrt(np.mean(e**2)), [i + 1], "o", color='darkred', label='RMSE')
    ticks.append("set-{}".format(i+1))

    if i == 0:
        plt.legend(loc='upper right', bbox_to_anchor=(.2,1.1))

plt.yticks(range(1, n+1, 1), ticks)
plt.xlabel('error')
ax.spines[['right', 'top']].set_visible(False)
plt.ylim(0, n + 1)
plt.xlim(-0.1, 1.1)

##############################################################################################################
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Define values and their corresponding probabilities
values = np.linspace(0,1,10)
p_normal = [0.01, 0.02, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.02, 0.01]  # These should sum up to 1
p_rshift = [0.01, 0.02, 0.05, 0.05, 0.1, 0.1, 0.3, 0.3, 0.02, 0.01]  # These should sum up to 1
p_routlier = [0.01, 0.02, 0.05, 0.15, 0.30, 0.05, 0.05, 0.05, 0.02, 0.25]  # These should sum up to 1
p_loutlier = p_routlier[::-1]  # These should sum up to 1

# Generate a biased set of random numbers
normal_set = random.choices(values, p_normal, k=1000)  # Change k to the desired number of samples\
rshift_set = random.choices(values, p_rshift, k=1000)  # Change k to the desired number of samples
routlier_set = random.choices(values, p_routlier, k=1000)  # Change k to the desired number of samples
loutlier_set = random.choices(values, p_loutlier, k=1000)  # Change k to the desired number of samples

# Create a 3x1 grid
grid = GridSpec(4, 1)

# Create subplots within the grid
plt.figure(figsize=(5, 5))

# Subplot 1
ax1 = plt.subplot(grid[0, 0])
plt.title('no bias')
ax1.spines[['right', 'top']].set_visible(False)
plt.hist(normal_set, color='grey')
mae = np.mean(normal_set)
plt.vlines(mae,0,500, linestyles='--', color='black', label='MAE: {:.3f}'.format(mae))
rmse = np.sqrt(np.mean(np.array(normal_set) ** 2))
plt.vlines(rmse, 0,500, linestyles='--', color='red', label = 'RMSE: {:.3f}'.format(rmse))
plt.legend(loc='upper left')

# Subplot 2
ax2 = plt.subplot(grid[1, 0])
plt.title('shift right')
ax2.spines[['right', 'top']].set_visible(False)
plt.hist(rshift_set, color='grey')
mae = np.mean(rshift_set)
plt.vlines(mae,0,500, linestyles='--', color='black', label='MAE: {:.3f}'.format(mae))
rmse = np.sqrt(np.mean(np.array(rshift_set) ** 2))
plt.vlines(rmse, 0,500, linestyles='--', color='red', label = 'RMSE: {:.3f}'.format(rmse))
plt.legend(loc='upper left')

# Subplot 3
ax3 = plt.subplot(grid[2, 0])
plt.title('outlier right')
ax3.spines[['right', 'top']].set_visible(False)
plt.hist(routlier_set, color='grey')
mae = np.mean(routlier_set)
plt.vlines(mae,0,500, linestyles='--', color='black', label='MAE: {:.3f}'.format(mae))
rmse = np.sqrt(np.mean(np.array(routlier_set) ** 2))
plt.vlines(rmse, 0,500, linestyles='--', color='red', label = 'RMSE: {:.3f}'.format(rmse))
plt.legend(loc='upper left')

# Subplot 4
ax4 = plt.subplot(grid[3, 0])
plt.title('outlier left')
ax3.spines[['right', 'top']].set_visible(False)
plt.hist(loutlier_set, color='grey')
mae = np.mean(loutlier_set)
plt.vlines(mae,0,500, linestyles='--', color='black', label='MAE: {:.3f}'.format(mae))
rmse = np.sqrt(np.mean(np.array(loutlier_set) ** 2))
plt.vlines(rmse, 0,500, linestyles='--', color='red', label = 'RMSE: {:.3f}'.format(rmse))
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

