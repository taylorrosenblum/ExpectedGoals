# generate an xG model using statsbomb events data filtered for "shots"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

# statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf

##############################################################################################################

from tqdm import tqdm  # Import tqdm for progress bars

# import table of shots, exclude headers
shots = pd.read_csv('shots_open_play.csv')

# transform table to shot_model format columns = ['Goal', 'X', 'Y', 'C', 'Dist', 'Angle']
shots_model = pd.DataFrame(columns=['Goal', 'X', 'Y'])

with tqdm(total = len(shots), desc="Fetching events data") as pbar:
    for i, row in shots.iterrows():

        # store relevant shot information in the shots_model data frame, note data references a 120 x 80 pitch
        shots_model.at[i, 'X'] = float(row['location'].strip('][').split(',')[0])
        shots_model.at[i, 'Y'] = float(row['location'].strip('][').split(',')[1])
        shots_model.at[i, 'C'] = abs(shots_model.at[i, 'Y'] - 40)
        shots_model.at[i, 'statsbomb_xg'] = eval(row['shot'])['statsbomb_xg']

        # determine goal boolean from outcome
        shots_model.at[i, 'Goal'] = False
        if eval(row['shot'])['outcome']['name'] == 'Goal':
            shots_model.at[i, 'Goal'] = True

        # calculate distance from goal
        x = 120 - shots_model.at[i, 'X']
        c = shots_model.at[i, 'C']
        shots_model.at[i, 'Distance'] = np.sqrt(x ** 2 + c ** 2)

        # calculate angle, how much of 180 deg in front of player contains the goal
        a = np.arctan(7.32 * x / (x ** 2 + c ** 2 - (7.32 / 2) ** 2))
        if a < 0:
            a = np.pi + a
        shots_model.at[i, 'Angle'] = a
        pbar.update(1)

# use statsmodels formula api to create a linear model to predict Goal from Distance and Angle
test_model = smf.glm(formula="Goal ~ Distance + Angle", data=shots_model, family=sm.families.Binomial()).fit()
p = test_model.params
print(test_model.summary())

# write model results to file
with open('model_summary.txt', 'w') as fh:
    fh.write(test_model.summary().as_text())

##############################################################################################################
# Model Vizualiztion

# create 2d array containing P_goal for XY locations using and model parameters "p"
pgoal_2d = np.zeros((120, 80))
for x in range(120):
    for y in range(80):
        a = np.arctan(7.32 * x /( x ** 2 + abs(y - 80 / 2) ** 2 - (7.32 / 2) ** 2))
        if a < 0:
            a = np.pi + a
        d = np.sqrt(x ** 2 + abs(y - 80 / 2) ** 2)
        pgoal_2d[x, y] = 1/(1+np.exp(p['Intercept'] + (p['Distance'] * d) + (p['Angle'] * a)))

# plot pitch using mplsoccer module
pitch = VerticalPitch(line_color='black', pitch_length=120, pitch_width=80, line_zorder = 2)
fig, ax = pitch.draw()

# plot probability on the pitch
pos = ax.imshow(pgoal_2d, cmap=plt.cm.plasma, vmin=0, vmax=0.3, zorder = 1)
fig.colorbar(pos, ax=ax)

# format the figure
ax.set_title('Multi-factor xG Model: Distance, Angle')
plt.xlim((0, 80))
plt.ylim((0, 80))
plt.gca().set_aspect('equal', adjustable='box')

# output some files
plt.savefig('xG_model.png')

##############################################################################################################
# Model Testing

# Mcfaddens Rsquared for Logistic regression
null_model = smf.glm(formula="Goal ~ 1 ", data=shots_model,
                     family=sm.families.Binomial()).fit()
1 - test_model.llf / null_model.llf

# ROC curve
numobs = 100
TP = np.zeros(numobs)
FP = np.zeros(numobs)
TN = np.zeros(numobs)
FN = np.zeros(numobs)

for i, threshold in enumerate(np.arange(0, 1, 1 / numobs)):
    for j, shot in shots_model.iterrows():
        if (shot['Goal'] == 1):
            if (shot['xG'] > threshold):
                TP[i] = TP[i] + 1
            else:
                FN[i] = FN[i] + 1
        if (shot['Goal'] == 0):
            if (shot['xG'] > threshold):
                FP[i] = FP[i] + 1
            else:
                TN[i] = TN[i] + 1

fig, ax = plt.subplots(num=1)
ax.plot(FP / (FP + TN), TP / (TP + FN), color='black')
ax.plot([0, 1], [0, 1], linestyle='dotted', color='black')
ax.set_ylabel("Predicted to score and did TP/(TP+FN))")
ax.set_xlabel("Predicted to score but didn't FP/(FP+TN)")
plt.ylim((0.00, 1.00))
plt.xlim((0.00, 1.00))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#fig.savefig('Output/ROC_' + model + '.pdf', dpi=None, bbox_inches="tight")


