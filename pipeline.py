""" Rubikloud take home problem """
import luigi
import csv
import re
import traceback
from collections import defaultdict

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
        with open(self.tweet_file, 'rU') as in_file:
            reader = csv.reader(in_file, delimiter=",", quoting=False)
            for row in reader:
                try:
                    tweet_id = row[17]
                    tweet_coord = row[15]
                    res = re.search('[\[]([-]?[0-9]+[.]?[0-9]+)[\s,]+([-]?[0-9]+[.]?[0-9]+)[\]]', tweet_coord)
                    if res is not None:
                        coordX = res.group(1)
                        coordY = res.group(2)
                        if coordX != 0 and coordY != 0:
                            row.extend(coordX, coordY)
                            data = data + [row]
                except:
                    traceback.print_exc()
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
        return CleanDataTask(tweet_file)

    def run(self):
        # clean it
        for t in self.input():
            with t.open('r') as in_file:
                for line in in_file:
        pass

class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')

    # TODO...


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')

    # TODO...


if __name__ == "__main__":
    luigi.run()
