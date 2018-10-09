from numpy.random import RandomState
import numpy as np


class Data:

    def get(self):
        raise NotImplementedError("method not implemented")

    def length(self):
        raise NotImplementedError("method not implemented")

    def name(self):
        raise NotImplementedError("method not implemented")

    def get_shape(self):
        raise NotImplementedError("method not implemented")

    def get_class_names(self):
        raise NotImplementedError("method not implemented")

    @staticmethod
    def gen_random_order(data_set_length):
        rand_state = RandomState(42)
        order = np.arange(0, data_set_length, 1)
        rand_state.shuffle(order)

        return order

    @staticmethod
    def translate_column(column, dataframe):
        values = dataframe[column].unique()
        value_to_label = dict(zip(range(len(values)), values))

        dataframe[column].replace(values, range(len(values)), inplace=True)

        return value_to_label


if __name__ == '__main__':
    d = Data()

    print d.gen_random_order(10)

    e = Data()

    print e.gen_random_order(10)