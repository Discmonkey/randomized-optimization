import sqlite3
from datasets.data import Data
import os
from datasets.top_level_file import base
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Fire(Data):

    def __init__(self, load_sql=True, normalize=True, balance=True):
        self.df_raw = None
        self.num_columns = None
        self.balance = balance
        self.value_dictionary = {}

        if load_sql:
            self.load_sql()

        if normalize:
            self.normalize()

    def load_sql(self, query=None):

        conn = sqlite3.connect(os.path.join(base, "raw/FPA_FOD_20170508.sqlite"))

        if query is None:
            query = """
                SELECT case FIRE_SIZE_CLASS WHEN 'G' THEN 1 WHEN 'F' THEN 1 WHEN 'E' THEN 1 ELSE 0 END FIRE_SIZE_CLASS,
                  cast(FIRE_YEAR as INTEGER) FIRE_YEAR,
                  cast(DISCOVERY_DOY as INTEGER) DISCOVERY_DOY,
                  cast(DISCOVERY_TIME as INT) DISCOVERY_TIME,
                  cast(LATITUDE as FLOAT) LAT,
                  cast(LONGITUDE as FLOAT) LNG
            """

            # flatten out the cause of the fire
            for i in range(1, 14):
                query += ", case STAT_CAUSE_CODE WHEN {} THEN 1 ELSE 0 END STAT_CAUSE_{}".format(i, i)

            query += """
                  FROM Fires;
            """

        self.df_raw = pd.read_sql_query(query, conn)

        if self.balance:
            self.df_raw = self.df_raw.groupby("FIRE_SIZE_CLASS")
            self.df_raw = self.df_raw.apply(
                lambda x: x.sample(min(1000, len(x))).reset_index(drop=True)
            )

        self.num_columns = len(self.df_raw.columns)

    def normalize(self):
        cols_to_norm = list(self.df_raw.columns)
        cols_to_norm.remove("FIRE_SIZE_CLASS")

        temp_df = self.df_raw[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        temp_df["FIRE_SIZE_CLASS"] = self.df_raw["FIRE_SIZE_CLASS"]

        self.df_raw = temp_df

        self.df_raw = self.df_raw[["FIRE_SIZE_CLASS"] + cols_to_norm]
        self.df_raw.fillna(0, inplace=True)

    def get(self):
        self.value_dictionary["FIRE_SIZE_CLASS"] = self.translate_column("FIRE_SIZE_CLASS", self.df_raw)

        return self.df_raw.values[:, 1:self.num_columns], self.df_raw.values[:, 0:1]

    def get_shape(self):
        return self.num_columns - 1, len(self.df_raw["FIRE_SIZE_CLASS"].unique())

    def length(self):
        return len(self.df_raw)

    def name(self):
        return "Fire"

    def plot_class_balance_pre(self):
        conn = sqlite3.connect(os.path.join(base, "raw/FPA_FOD_20170508.sqlite"))
        df = pd.read_sql("SELECT count(*) [Number of Fires], sum(FIRE_SIZE) [Total Acres Burned], "
                         "FIRE_SIZE_CLASS Class FROM fires GROUP BY FIRE_SIZE_CLASS", conn)

        df.plot.bar(x="Class", subplots=True)
        plt.show()

    def plot_class_balance_post(self):
        pass

    def save(self):
        train, test = train_test_split(self.df_raw, test_size=.1, random_state=100)
        train.to_csv(base + "/cache/train_fire_reduced.csv", index=False)
        test.to_csv(base + "/cache/test_fire_reduced.csv", index=False)


if __name__ == '__main__':
    f = Fire(normalize=True)
    f.save()

