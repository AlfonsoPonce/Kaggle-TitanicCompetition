import pandas as pd
#from pandasgui import show

class Analysis:

    def __init__(self, path):
        self.path = path

    def openFile(self):
        return pd.read_csv(self.path)

    #def opeFileGUI(self):
    #    show(self.path)

    #Death rates from gender
    def femaleVsMale(self, data):
        total_people = len(data[:])

        male_people = len(data[data['Sex'] == 'male'])
        female_people = total_people - male_people


        male_rate = round((male_people / total_people) * 100)
        female_rate = round((female_people / total_people) * 100)
        P_M = 0.65
        P_F = 0.35


        print(str(male_rate) + '% of the passengers where male')
        print(str(female_rate) + '% of the passengers where female')


        survivors = len(data[data['Survived'] == 1])
        survivor_rate = round((survivors / total_people) * 100)



