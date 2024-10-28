import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

class SelectColumns(BaseEstimator, TransformerMixin):
	def __init__(self, columns):
		self.columns = columns

	def fit(self, xs, ys, **params):
		return self

	def transform(self, xs):
		return xs[self.columns]

def random_forest(xs, ys):
	print('Random forest:')
	steps = [('column_select', SelectColumns(xs.columns)), ('random_forest_regressor', RandomForestRegressor(random_state=42))]
	pipe = Pipeline(steps)
	grid = {'column_select__columns': [list(xs.columns[0:index]) for index in range(1, 50, 25)], 'random_forest_regressor__n_estimators': [50, 100, 150], 'random_forest_regressor__max_depth': [None, 10, 20, 30]}
	search = GridSearchCV(pipe, grid, scoring = 'r2', n_jobs = -1)
	search.fit(xs, ys)
	print('R-squared: ' + str(search.best_score_))
	print('Best params: ' + str(search.best_params_) + '\n')

def linear_regression(xs, ys):
	print('LinearRegression:')
	steps = [('column_select', SelectColumns(xs.columns)), ('linear_regression', LinearRegression(n_jobs = -1))]
	pipe = Pipeline(steps)
	grid = {'column_select__columns': [list(xs.columns[0:index]) for index in range(1, 250, 10)], 'linear_regression': [LinearRegression(n_jobs = -1), TransformedTargetRegressor(LinearRegression(n_jobs = -1), func = np.sqrt, inverse_func = np.square), TransformedTargetRegressor(LinearRegression(n_jobs = -1), func = np.cbrt, inverse_func = lambda y: np.power(y, 3)), TransformedTargetRegressor(LinearRegression(n_jobs = -1), func = np.log, inverse_func = np.exp)]}
	search = GridSearchCV(pipe, grid, scoring = 'r2', n_jobs = -1)
	search.fit(xs, ys)
	print('R-squared: ' + str(search.best_score_))
	print('Best params: ' + str(search.best_params_) + '\n')

def gradient_boosting(xs, ys):
	print('Gradient boosting:')
	steps = [('column_select', SelectColumns(xs.columns)), ('gradient_boosting_regressor', GradientBoostingRegressor(random_state=42))]
	pipe = Pipeline(steps)
	grid = {'column_select__columns': [list(xs.columns[0:index]) for index in range(1, 50, 25)], 'gradient_boosting_regressor__n_estimators': [50, 100, 150], 'gradient_boosting_regressor__learning_rate': [0.01, 0.1, 0.2]}
	search = GridSearchCV(pipe, grid, scoring = 'r2', n_jobs = -1)
	search.fit(xs, ys)
	print('R-squared: ' + str(search.best_score_))
	print('Best params: ' + str(search.best_params_) + '\n')

def decision_tree(xs, ys):
	print('Decision tree:')
	steps = [('column_select', SelectColumns(xs.columns)), ('decision_tree_regressor', DecisionTreeRegressor(random_state=42))]
	pipe = Pipeline(steps)
	grid = {'column_select__columns': [list(xs.columns[0:index]) for index in range(1, 50, 25)], 'decision_tree_regressor__max_depth': [None, 5, 10, 15], 'decision_tree_regressor__min_samples_split': [2, 5, 10]}
	search = GridSearchCV(pipe, grid, scoring = 'r2', n_jobs = -1)
	search.fit(xs, ys)
	print('R-squared: ' + str(search.best_score_))
	print('Best params: ' + str(search.best_params_) + '\n')

def main():
	data = pandas.read_csv('AmesHousing.csv')
	data = data.drop(columns = ['Neighborhood']) # drop neighborhood as it's not to be used
	data = pandas.get_dummies(data)
	for column in data.columns:
		data[column].fillna(data[column].mean(), inplace = True) # soon to be deprecated (NOT AN ERROR)
	xs = data.drop(columns = ['SalePrice'])
	ys = data['SalePrice']
	random_forest(xs, ys)
	linear_regression(xs, ys)
	gradient_boosting(xs, ys)
	decision_tree(xs, ys)

if __name__ == '__main__':
	main()