import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from typing import List, Union
from scipy import stats

class Preprocessing:
    def __init__(self, data):
        self.data = data

    # Names and Ids are not relevant
    def eraseUselessColumns(self):
        """ Erase of useless columns

        :return:
        pandas_dataframe: data frame with erased columns
        """
        self.data.drop('Name', axis=1, inplace=True)
        self.data.drop('PassengerId', axis=1, inplace=True)
        self.data.drop('Ticket', axis=1, inplace=True)
        self.data.drop('Cabin', axis=1, inplace=True)
        self.data.drop('Embarked', axis=1, inplace=True)
        self.data.drop('Fare', axis=1, inplace=True)
        return self.data

    def detectColumnsWithNull(self):
        """ Get columns that have null values

        :return: string array: list of columns that have null values
        """
        columns = []
        for i in self.data.columns:
            bool_series = pd.isnull(self.data[i])
            has_null = False
            for j in range(len(bool_series)):
                has_null = has_null or bool_series.iloc[j]
            if has_null:
                columns.append(i)

        return columns

    def percentageOfMissingValues(self, columns):
        """ Relative percentage of missing values for each column. Prints those percentages.

        :param: columns(string array): columns with null values
        :return:
        """
        total_people = len(self.data[:])

        for c in columns:
            array = self.data[c]
            num_nulls = array.isna().sum()
            print("Column " + c +" has " + str(round(num_nulls / total_people * 100)) + "% of MVs")

    def MissingValuesAnalysis(self, missing_column, column_to_compare):
        """

        :param missing_column:
        :param column_to_compare:
        :return:
        """
        null_column_data = self.data[self.data[missing_column].isnull()]
        to_compare_data = null_column_data[column_to_compare]
        total_nan = len(to_compare_data)

        if column_to_compare == "Sex":
            v1 = "male"
            v2 = "female"
        elif column_to_compare == "Survived":
            v1 = 0
            v2 = 1
        elif column_to_compare == "Pclass":
            v1 = 1
            v2 = 2
            v3 = 3
            total_v3 = len(self.data[self.data[column_to_compare] == v3])
            num_v1 = 0
            num_v2 = 0
            num_v3 = 0
            for i in range(len(to_compare_data)):
                if to_compare_data.iloc[i] == v1:
                    num_v1 += 1
                elif to_compare_data.iloc[i] == v2:
                    num_v2 += 1
                else:
                    num_v3 += 1
        total_v1 = len(self.data[self.data[column_to_compare] == v1])

        total_v2 = len(self.data[self.data[column_to_compare] == v2])

        if column_to_compare != "Pclass":
            num_v1 = 0
            for i in range(len(to_compare_data)):
                if to_compare_data.iloc[i] == v1:
                    num_v1 += 1
            num_v2 = len(to_compare_data) - num_v1

        print(str(num_v1 / total_v1 * 100) + '%')
        print(str(num_v2 / total_v2 * 100) + '%')
        if column_to_compare == "Pclass":
            print(str(num_v3 / total_v3 * 100) + '%')
            p2 = plt.bar(x=[str(v1), str(v2), str(v3)], height=[total_v1, total_v2, total_v3])
            p1 = plt.bar(x=[str(v1), str(v2), str(v3)], height=[num_v1, num_v2, num_v3])
        else:
            p2 = plt.bar(x=[str(v1), str(v2)], height=[total_v1, total_v2])
            p1 = plt.bar(x=[str(v1), str(v2)], height=[num_v1, num_v2])
       # p2 = plt.bar(x=[str(v1), str(v2)], height=[total_v1, total_v2])
        plt.title("Num of NaN Age grouped by " + column_to_compare)
        plt.legend((p1[0], p2[0]), (column_to_compare + ' w/ NaN Age', 'total '+column_to_compare+' count'))
        plt.show()


    def pieChart(self, reference_column, value, column_to_measure):
        """

        :param reference_column:
        :param value:
        :param column_to_measure:
        :return:
        """
        total = len(self.data[self.data[reference_column] == value])

        if column_to_measure == "Pclass":
            data_1 = self.data[self.data[column_to_measure] == 1]
            data_2 = self.data[self.data[column_to_measure] == 2]
            data_3 = self.data[self.data[column_to_measure] == 3]

            n_1 = len(data_1[data_1[reference_column] == value])
            n_2 = len(data_2[data_2[reference_column] == value])
            n_3 = len(data_3[data_3[reference_column] == value])

            plt.pie([n_1, n_2, n_3], labels=['class 1', 'class 2', 'class 3'],autopct='%1.1f%%')
            plt.title("Dead percentages by class")
            plt.show()

    def twoPlot(self, x_dim, y_dim):
        x = self.data[x_dim]
        y = self.data[y_dim]

        plt.scatter(x, y, marker="o")
        plt.show()

    #¿Que clases a tener en cuenta para agrupar?
    def imputeCMC(self, column_to_impute, condition_columns):
        columns = list(self.data.columns)
        columns.remove(column_to_impute)
        self.data[column_to_impute] = self.data[column_to_impute].fillna(self.data.groupby(condition_columns)[column_to_impute].transform("mean"))

        return self.data


    def stringTransformation(self):
        self.data.loc[self.data["Sex"] == "male", "Sex"] = 1
        self.data.loc[self.data["Sex"] == "female", "Sex"] = 2

        return self.data



    def corr4ContinuousVsCategorical(self, continuous_col, categorical_col, method):
        new_df = self.data[[continuous_col, categorical_col]]
        if method == 'variance':
            cont_variance = np.var(new_df[continuous_col])
            print("Overall variance = " + str(cont_variance))
            print("------------------------------")
            for category in new_df[categorical_col].unique():
                all_fares = new_df[new_df[categorical_col] == category]
                grouped_variance = np.var(all_fares)
                print(str(category) + " class has a variance of " + str(grouped_variance['Fare']))
        else:
            print('Method not implemented')

    def Normalize(self, columns, method):
        if method == 'MinMax':
            for c in columns:
                self.data[c] = (self.data[c] - self.data[c].min()) / (self.data[c].max() - self.data[c].min())

            return self.data

        elif method == "ZScore":
            for c in columns:
                self.data[c] = (self.data[c] - self.data[c].mean()) / (self.data[c].std())

            return self.data

        elif method == "BoxCox":
            for c in columns:
                self.data[c],_ = stats.boxcox(self.data[c])
            return self.data
        elif method == "Log":
            for c in columns:
                self.data[c] = np.log(self.data[c])
            return self.data
        elif method == "Pow":
            for c in columns:
                self.data[c] = np.power(self.data[c],2)
            return self.data
        else:
            print('Not implemented')

    def bin_ages(self, age_series: pd.Series) -> pd.Series:
        age_labels = [f"[{i}, {i + 10})" for i in range(0, 91, 10)]

        age_bins = pd.IntervalIndex.from_tuples(
            [(i, i + 10) for i in range(0, 91, 10)],
            closed="left"
        )

        ages_binned = pd.cut(
            age_series,
            age_bins,
            labels=age_labels,
            precision=0,
            include_lowest=True
        )

        ages_binned.sort_values(ascending=True, inplace=True)
        # Change the values from categorical to string to be able to plot them
        ages_binned = ages_binned.astype("str")

        return ages_binned

    def plot_histogram(self, data_series: pd.Series, nbins: int, title: str, axes_titles: List[Union[str, None]]) -> None:
        fig = px.histogram(x=data_series, nbins=nbins,title=title)

        fig.update_layout(xaxis_title=axes_titles[0], yaxis_title=axes_titles[1])

        fig.update_layout(uniformtext_minsize=14, uniformtext_mode="hide", bargap=0, title_x=0.5)

        fig.show()

    def discreteHistogram(self, series):
        data_binned = self.bin_ages(series)

        self.plot_histogram(data_binned, 10, "Ages binned", ["Ages", None])

    def roundingAges(self):
        self.data['Age'] = self.data['Age'].apply(np.ceil)

        return self.data

    def boxPlot(self, column):
        fig1, ax1 = plt.subplots()
        ax1.set_title('Basic Plot')
        ax1.boxplot(self.data[column])
        plt.show()





