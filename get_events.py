# create a dataframe of shots from statsbomb events data

from statsbombpy import sb
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars


# Add a print statement to indicate the program has started
print("Fetching data from statsbomb and processing...")

# index for competition and season id's, id's used to query statsbomb data

''''
index = {0 : {'name': 'Premier League 15/16', 'competition': 2, 'season': 27},
         1 : {'name': 'Ligue 1 15/16', 'competition': 7, 'season': 27},
         2 : {'name': 'La Liga 15/16', 'competition': 11, 'season': 27},
         3 : {'name': '1. Bundesliga 15/16', 'competition': 9, 'season': 27},
         4 : {'name': 'Serie A 15/16', 'competition': 12, 'season': 27}}
'''
index = {0 : {'name': '22MWC', 'competition': 43, 'season': 106}}

# for some know comp / seasons, get list of match id_s
match_ids = []
for i in index:
    ids = sb.matches(competition_id=index[i]['competition'], season_id=index[i]['season'])['match_id']
    match_ids.extend(ids)

# iterate through list of match_ids, query for events data for that given match, filter for "shot" events
# concatenate all shot events to single dataframe
with tqdm(total = len(match_ids), desc="Fetching events data") as pbar:
    shots_data = pd.DataFrame()

    for id in match_ids:
        events = sb.events(match_id=id, split=True, flatten_attrs=False)['shots']
        shots_data = pd.concat([shots_data, events])
        pbar.update(1)

# Add a print statement to indicate the program has finished
print("Data fetching and processing completed.")

# save dataframe locally as csv
shots_data.to_csv('test_data.csv')

# print statement to indicate the file has been saved
print("Data saved as test_data.csv")