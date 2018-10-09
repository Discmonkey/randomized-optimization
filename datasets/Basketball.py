from datasets.data import Data
import pandas as pd
import numpy as np
from top_level_file import base
from sklearn.model_selection import train_test_split


class BasketBall(Data):

    def __init__(self, normalize=True):
        self.df_raw = pd.DataFrame.from_csv(base + "/" + "raw/shot_logs.csv")
        self.df_processed = None
        self.num_columns = None
        self.translation_dict = {}
        self.process()
        self.is_normalized = normalize

        self.x = self.df_processed.values[:, 1:self.num_columns]
        self.y = self.df_processed.values[:, 0:1]

        if normalize:
            self.normalize()

    def name(self):
        return "Basketball"

    def normalize(self):
        cols_to_norm = list(self.df_processed.columns)
        cols_to_norm.remove("TEAM")

        temp_df = self.df_processed[cols_to_norm].apply(lambda x: ((x - x.min()) / (x.max() - x.min())) - .5)
        temp_df["TEAM"] = self.df_processed["TEAM"]

        self.df_processed = temp_df

        self.df_processed = self.df_processed[["TEAM"] + cols_to_norm]
        self.df_processed.fillna(0, inplace=True)

    def calculate_seconds_left_in_game(self):
        self.df_raw['SECONDS_IN_GAME'] = self.df_raw.GAME_CLOCK.str.extract('(\d{1,2}):').astype('int') * 60 + \
                                         self.df_raw.GAME_CLOCK.str.extract(':(\d{1,2})').astype('int')

    def process(self):
        """
            This method collapses the individual shot log data into a breakdown of summary data by team and quarter

        :return:
        """
        # look at statistics for specific quarters [1, 2, 3, 4]

        # now let's get the time in seconds for the quarter
        self.df_raw['TEAM'] = self.df_raw.MATCHUP.str.extract('.* - (.{3})')

        # figure out if the point was a threepointer as it's own column
        self.df_raw['IS_THREE'] = (self.df_raw.PTS_TYPE == 3).astype(int)
        self.df_raw['IS_TWO'] = (self.df_raw.PTS_TYPE == 2).astype(int)

        self.df_raw = self.df_raw[['TEAM', 'PERIOD', 'MATCHUP', 'SHOT_CLOCK', 'DRIBBLES',
                                   'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'FGM',
                                   'CLOSE_DEF_DIST', 'IS_THREE', 'IS_TWO']]

        grouped_obj = self.df_raw.groupby(['TEAM', 'MATCHUP'])

        num_shots_frame = grouped_obj.size().to_frame(name='NUM_SHOTS')

        self.df_processed = num_shots_frame.join(
            grouped_obj.aggregate({'SHOT_CLOCK': np.mean, 'DRIBBLES': np.mean,
                                   'TOUCH_TIME': np.mean, 'SHOT_DIST': np.mean, 'CLOSE_DEF_DIST': np.mean,
                                   'IS_THREE': np.sum, 'IS_TWO': np.sum, 'FGM': np.sum})
                       .rename(columns={'SHOT_CLOCK': 'AVG_SHOT_CLOCK',
                                        'DRIBBLES': 'AVG_DRIBBLES',
                                        'TOUCH_TIME': 'AVG_TOUCH_TIME',
                                        'SHOT_DIST': 'AVG_SHOT_DIST',
                                        'IS_THREE': 'THREES_TAKEN',
                                        'IS_TWO': 'TWOS_TAKEN', 'CLOSE_DEF_DIST': 'AVG_CLOSEST_DEFENDER'})
        ).reset_index()[['TEAM', 'NUM_SHOTS', 'FGM', 'THREES_TAKEN', 'TWOS_TAKEN', 'AVG_TOUCH_TIME',
                        'AVG_DRIBBLES', 'AVG_SHOT_DIST', 'AVG_CLOSEST_DEFENDER', 'AVG_SHOT_CLOCK']]

        self.num_columns = len(self.df_processed.columns)

        for column in ['TEAM']:
            values = self.df_processed[column].unique()
            value_to_label = dict(zip(range(len(values)), values))

            self.translation_dict[column] = value_to_label

            self.df_processed[column].replace(values, range(len(values)), inplace=True)

        return 0

    def get_shape(self):
        return self.num_columns - 1, len(self.df_raw['TEAM'].unique())

    def get(self):
        return self.df_processed.values[:, 1:self.num_columns], self.df_processed.values[:, 0:1]

    def save(self):
        train, test = train_test_split(self.df_processed, test_size=.1, random_state=100)
        train.to_csv(base + "/cache/train_basketball_no_period.csv", index=False)
        test.to_csv(base + "/cache/test_basketball_no_period.csv", index=False)


if __name__ == '__main__':
    instance = BasketBall()
    instance.save()
