""" Rubikloud take home problem """
import luigi
import csv
import re
import traceback
from collections import defaultdict
import operator
import pandas as pd
from sklearn import tree
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        data = []
        with open(self.tweet_file, 'rU') as in_file: # standard 'r' option does not work with my system
            reader = csv.reader(in_file, delimiter=",", quoting=False)
            for row in reader:
                tweet_id = row[17]
                tweet_coord = row[15]
                res = re.search('[\[]([-]?[0-9]+[.]?[0-9]+)[\s,]+([-]?[0-9]+[.]?[0-9]+)[\]]', tweet_coord) # using regex to filter tweet_coord
                if res is not None:
                    coordX = float(res.group(1))
                    coordY = float(res.group(2))
                    if coordX != 0 and coordY != 0:
                        row.extend([coordX, coordY])
                        data = data + [row]
        with self.output().open('w') as out_file:
            writer = csv.writer(out_file, delimiter=",")
            for row in data:
                writer.writerow(row)

class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')

    def output(self):
        return luigi.LocalTarget(self.output_file)
    
    def requires(self):
        return CleanDataTask(self.tweet_file)

    def run(self):
        cities = pd.read_csv(self.cities_file, delimiter=",")
        data = []

        with self.input().open('r') as in_file:
            reader = csv.reader(in_file, delimiter=",", quoting=False)
            for row in reader:
                # determine the X (city) value
                coordX = row[-2]
                coordY = row[-1]
                xdiffs = [(float(coordX) - float(x))**2 for x in cities['latitude'].tolist()]
                ydiffs = [(float(coordY) - float(y))**2 for y in cities['longitude'].tolist()]
                l2 = [xdiffs[i] + ydiffs[i] for i in range(len(xdiffs))] # Iterating over entire city list
                
                min_index, min_value = min(enumerate(l2), key=operator.itemgetter(1)) # Find closest city
                city = cities.loc[min_index]['asciiname']
                # determine the Y (sentiment) value
                airline_sentiment = row[5]
                sentiment = 2
                if airline_sentiment == 'neutral':
                    sentiment = 1
                elif airline_sentiment == 'negative':
                    sentiment = 0
                row.extend([city, sentiment])
                data = data + [row]                        
    
        with self.output().open('w') as out_file:
            writer = csv.writer(out_file, delimiter=",")
            for row in data:
                writer.writerow(row)


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')
    cities_file = luigi.Parameter(default='cities.csv')

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def requires(self):
        return TrainingDataTask(self.tweet_file)
    
    def run(self):
        X = []
        y = []
        cities = pd.read_csv(self.cities_file, delimiter=",")

        with self.input().open('r') as in_file:
            try:
                reader = csv.reader(in_file, delimiter=",", quoting=False)
                for row in reader:
                    X = X + [row[-2]]
                    y= y + [int(row[-1])]
                enc = OneHotEncoder()
                le = LabelEncoder()
                le.fit(cities['asciiname'].tolist())
                new_X = le.transform(X) # Replace all cities in training data with labels
                all_cities = le.transform(cities['asciiname'].tolist()) 
                enc.fit(all_cities.reshape(-1, 1)) # Fit one-hot transformation scheme with full city list
                X_one_hot = enc.transform(new_X.reshape(-1, 1)) # Transform training data city labels to one-hot vectors
                clf = tree.DecisionTreeClassifier() # Decision tree model
                clf = clf.fit(X_one_hot, y)
            except:
                traceback.print_exc() 
                
        with self.output().open('w') as out_file:
            pickle.dump(clf, out_file)

class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='scores.csv')

    def output(self):
        return luigi.LocalTarget(self.output_file)
    
    def requires(self):
        return TrainModelTask(self.tweet_file)
    
    def run(self):
        # generate cities to be scored
        cities = pd.read_csv(self.cities_file, delimiter=",")

        with self.input().open('r') as in_file:
            clf = pickle.load(in_file)
            X = cities['asciiname'].tolist()
            enc = OneHotEncoder()
            le = LabelEncoder()
            le.fit(X)
            new_X = le.transform(X)
            enc.fit(new_X.reshape(-1, 1))
            X_one_hot = enc.transform(new_X.reshape(-1, 1))
            predictions = clf.predict(X_one_hot)
            probs = clf.predict_proba(X_one_hot)

        with self.output().open('w') as out_file:
            df = pd.DataFrame(probs, columns=['negative', 'neutral', 'positive'])
            df['city'] = cities['asciiname']
            print >> out_file, df.to_csv(sep=',', header=True, index=False) 
            
if __name__ == "__main__":
    luigi.run()
