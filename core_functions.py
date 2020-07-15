import pandas as pd
from helper_functions import optimize_memory
import numpy as np
from scipy import stats
import pickle as p

# from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

import sys, json, requests
import operator
import random
import base64
import zlib
import os
from pathlib import Path
import gc
import shutil
import os
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
log = logging.getLogger('fillet-flask.sub')





# class GA(object):

# 	#initialize variables and lists
# 	def __init__(self): 

# 		self.models = []
# 		self.price_std = []
# 		self.price_mean = []
# 		self.parents = []
# 		self.newparents = []
# 		self.bests = []
# 		self.best_p = [] 
# 		self.price_names = []
# 		self.iterated = 1
# 		self.population = 0
# 		self.epoch = 0
# 		self.best_price = []

# 		# increase max recursion for long stack
# 		iMaxStackSize = 15000
# 		sys.setrecursionlimit(iMaxStackSize)

# 	# create the initial population 
# 	def initialize(self):
# 		self.num_item = self.price_mean.shape[0]
# 		for i in range(self.population):
# 			parent = (np.random.standard_normal(self.num_item)*self.price_std*2 + self.price_mean).tolist()
# 			self.parents.append(parent)

# 	# set the details of this problem
# 	def properties(self, models, population, max_epoch, price_std, price_mean, price_names):

# 		self.models = models
# 		self.population = population
# 		self.epoch = max_epoch
# 		self.price_std = price_std
# 		self.price_mean = price_mean
# 		self.price_names = price_names
# 		self.initialize()

# 	# calculate the fitness function for X
# 	def fitness(self, prices):
		
# 		df = pd.DataFrame(prices, columns = self.price_names) 
# 		quantities = np.zeros((self.population, len(self.models)))
# 		for i in range(len(self.models)):
# 			quantities[:, i] = self.models[i].predict(df)
# 		fitness = np.zeros((self.population))
# 		for i in range(len(self.models)):
# 			fitness += prices[:, i]*quantities[:, i]
		
# 		return fitness
   
# 	# run generations of GA
# 	def evaluation(self):

# 		# loop through parents and calculate fitness
# 		best_pop = self.population // 2
# 		ft = self.fitness(np.asarray(self.parents))
# 		for i in range(len(self.parents)):
# 			parent = self.parents[i]
# 			self.bests.append((ft[i], parent))

# 		# sort the fitness list by fitness
# 		self.bests.sort(key=operator.itemgetter(0), reverse=True)
# 		self.best_p = self.bests[:best_pop]
# 		highest_fitness = self.best_p[0][0]
# 		highest_price = self.best_p[0][1]
# 		self.best_p = [x[1] for x in self.best_p]
		
# 		return highest_fitness, highest_price

# 	# mutate children after certain condition
# 	def mutation(self, ch):

# 		for i in range(len(ch)):
# 			if random.uniform(0, 1) > 0.95:
# 				ch[i] = np.random.standard_normal(1)[0]*self.price_std[i]*2 + self.price_mean[i]
# 		return ch

# 	# crossover two parents to produce two children by miixing them under random ration each time
# 	def crossover(self, ch1, ch2):

# 		threshold = random.randint(1, len(ch1)-1)
# 		tmp1 = ch1[threshold:]
# 		tmp2 = ch2[threshold:]
# 		ch1 = ch1[:threshold]
# 		ch2 = ch2[:threshold]
# 		ch1.extend(tmp2)
# 		ch2.extend(tmp1)

# 		return ch1, ch2

# 	# run the GA algorithm
# 	def run(self):
# 		if self.epoch < self.iterated:
# 			return self.best_price
# 		# run the evaluation once
# 		highest_fitness, highest_price = self.evaluation()
# 		newparents = []
# 		pop = len(self.best_p)
		
# 		# print("{}th generation" .format(self.iterated))
# 		# print("best solution so far: {}".format(highest_fitness))
# 		self.best_price = highest_price

# 		# randomly shuffle the best parents
# 		random.shuffle(self.best_p)
# 		for i in range(0, pop):
# 			if i < pop-1:
# 				r1 = self.best_p[i]
# 				r2 = self.best_p[i+1]
# 				nchild1, nchild2 = self.crossover(r1, r2)
# 				newparents.append(nchild1)
# 				newparents.append(nchild2)
# 			else:
# 				r1 = self.best_p[i]
# 				r2 = self.best_p[0]
# 				nchild1, nchild2 = self.crossover(r1, r2)
# 				newparents.append(nchild1)
# 				newparents.append(nchild2)

# 		# mutate the new children and potential parents to ensure global optima found
# 		for i in range(len(newparents)):
# 			newparents[i] = self.mutation(newparents[i])	

# 		self.parents = newparents
# 		self.bests = []
# 		self.best_p = []
# 		self.iterated += 1
# 		return self.run()




class rms_pricing_model():
	def __init__(self, data):
		
		self.models = {}
		
		# Ingest Correctly Shaped Data
		sales_data = data[['Wk', 'Tier', 'Store', 'Item_ID', 'Qty_', 'Price_']].copy()

		del data
		gc.collect()

		# Optimize Memory Usage by Downcasting etc.
		sales_data = optimize_memory(sales_data)

		# Convert Data to Wide Format
		sales_data_wide = sales_data.set_index(
			['Wk', 'Tier', 'Store',
			 'Item_ID']).unstack(level=-1).reset_index().copy()

		del sales_data
		gc.collect()

		sales_data_wide.columns = [
			''.join(str(i) for i in col).strip()
			for col in sales_data_wide.columns.values
		]
		sales_data_wide = sales_data_wide.sort_values(
			['Tier', 'Store', 'Wk'], ascending=True).reset_index(drop=True)

		# # Remove Store,Weeks with Nan Sales for Some Items ( Not sure if we want to do this or replace with 0 )
		# # Drops about 421 rows, hence seems reasonable to remove
		sales_data_wide_clean = sales_data_wide.dropna(axis=0).copy()

		del sales_data_wide
		gc.collect()

		self.data = sales_data_wide_clean.copy()
		self.price_columns = [
		col for col in sales_data_wide_clean.columns if col.startswith('Price')
		]

		del sales_data_wide_clean
		gc.collect()

	def get_and_save_price_info(self, price_info_path):
		prices = self.data[self.price_columns]
		prices_std = prices.std(axis = 0, skipna = True)
		prices_mean = prices.mean(axis = 0, skipna = True)
		p.dump((prices_std, prices_mean, self.price_columns), open(price_info_path,'wb'))


	def get_performance(self, item_id, proj_id):
		
		log.info(f'CROSS VALIDATING ITEM_ID {item_id} MODEL')

		# with open('keys.json') as f:
		# 	HOST_KEY = json.load(f)['host_key']

		HOST_KEY = os.environ['FUNCTIONS_KEY']                                         	

		sales_data_wide_clean = self.data.copy()

		target_colname = 'Qty_' + str(item_id)

		if target_colname not in sales_data_wide_clean.columns:
			print('Item Not Found in Dataset.')
			return None

		Price_columns = [
			col for col in sales_data_wide_clean.columns
			if col.startswith('Price')
		]
		target_column = [target_colname]

		Week = sales_data_wide_clean[['Wk']].copy()
		X = sales_data_wide_clean[Price_columns].copy()
		X = X.reindex(sorted(X.columns), axis=1)
		y = sales_data_wide_clean[target_column].copy()

		payload = {
		'code':HOST_KEY,
		}

		temp_cv_path = f'temp/cv/{proj_id}/{item_id}'

		if not os.path.isdir(temp_cv_path):
			Path(temp_cv_path).mkdir(parents=True)

		while True:
			try:

				X.to_parquet(temp_cv_path+'/X.parquet')
				y.to_parquet(temp_cv_path+'/y.parquet')
				Week.to_parquet(temp_cv_path+'/Wk.parquet')

				

				files = {'X_file': open(temp_cv_path+'/X.parquet', 'rb'),
						 'y_file': open(temp_cv_path+'/y.parquet', 'rb'),
						 'Wk_file': open(temp_cv_path+'/Wk.parquet', 'rb'),
						 }

				url = 'https://sutdcapstone22-filletofish.azurewebsites.net/api/fillet_func_2_cv'
				# url = 'http://localhost:7071/api/fillet_func_2_cv'
				result = requests.get(
					url, params=payload,
					files=files
					)
				outp = result.json()
				outp['item_id'] = int(item_id)
				break
			except Exception as e:
				log.info(f'CV {item_id} FAILED. RETRYING...')
				log.info(f'{e}')
				time.sleep(60)
				pass

		del sales_data_wide_clean
		gc.collect()

		HOME = os.environ['HOME_SITE']
		# HOME = ''

		cv_results_path = HOME+f'/projects/{proj_id}/cv/'
		if not os.path.isdir(cv_results_path):
			Path(cv_results_path).mkdir(parents=True)

		with open(cv_results_path+f'{item_id}_cv_perf.json', 'w') as outfile:
			# json.dump(result.json(), outfile)
			json.dump(outp, outfile)

		files['X_file'].close()
		files['y_file'].close()
		files['Wk_file'].close()

		del X
		del y
		del Week
		gc.collect()

		return outp

	def get_all_performance(self, proj_id):
		item_ids = [int(x.split('_')[1]) for x in self.price_columns]
		perf_df = pd.DataFrame(columns=[
			'item_id','avg_sales','r2_score',
			'mae_score','mpe_score','rmse_score']
			)

		processes_cv = []
		results_ls_cv = []
		with ThreadPoolExecutor(max_workers=5) as executor:
			for item_id in item_ids:
				processes_cv.append(executor.submit(self.get_performance,item_id,proj_id))
		for task in as_completed(processes_cv):
			perf_df = perf_df.append(task.result(), ignore_index=True)

		shutil.rmtree('temp/cv')

		# Check That All Models are Trained, Else Retrain

		HOME = os.environ['HOME_SITE']
		# HOME = ''

		proj_cv_path = HOME+f'/projects/{proj_id}/cv'
		cv_filenames = os.listdir(proj_cv_path)
		cv_models = [int(x.split('_')[0]) for x in cv_filenames]

		for item_id in item_ids:
			processes_incomplete = []
			results_ls_incomplete = []
			with ThreadPoolExecutor(max_workers=5) as executor:
				if item_id not in cv_models:
					# Retrain and Re-Save to Disk
					processes_incomplete.append(executor.submit(self.get_performance,item_id,proj_id))
			for task in as_completed(processes_incomplete):
				results_ls_incomplete.append(task.result())

		return perf_df

	def get_model(self, item_id, proj_id):

		log.info(f'TRAINING ITEM_ID {item_id} MODEL')

		# with open('keys.json') as f:
		# 	HOST_KEY = json.load(f)['host_key']

		HOST_KEY = os.environ['FUNCTIONS_KEY']                                         	

		sales_data_wide_clean = self.data.copy()

		target_colname = 'Qty_' + str(item_id)

		if target_colname not in sales_data_wide_clean.columns:
			print('Item Not Found in Dataset.')
			return None

		Price_columns = [
			col for col in sales_data_wide_clean.columns
			if col.startswith('Price')
		]
		target_column = [target_colname]

		X = sales_data_wide_clean[Price_columns].copy()
		X = X.reindex(sorted(X.columns), axis=1)
		y = sales_data_wide_clean[target_column].copy()

		del sales_data_wide_clean
		gc.collect()

		payload = {
		'code':HOST_KEY,
		}

		temp_train_path = f'temp/train/{proj_id}/{item_id}'

		if not os.path.isdir(temp_train_path):
			Path(temp_train_path).mkdir(parents=True)


		
		while True:
			try:
				X.to_parquet(temp_train_path+'/X.parquet')
				y.to_parquet(temp_train_path+'/y.parquet')

				files = {'X_file': open(temp_train_path+'/X.parquet', 'rb'),
						 'y_file': open(temp_train_path+'/y.parquet', 'rb')}

				url = 'https://sutdcapstone22-filletofish.azurewebsites.net/api/fillet_func_1_train'
				# url = 'http://localhost:7071/api/fillet_func_1_train'
				result = requests.get(url, params=payload,
					files=files
					)
				model = p.loads(result.content)
				break

			except Exception as e:
				log.info(f'TRAIN {item_id} FAILED. RETRYING...')
				log.info(f'{e}')
				time.sleep(60)
				pass


		model = p.loads(result.content)

		del X
		del y
		gc.collect()

		self.models[item_id] = model

		HOME = os.environ['HOME_SITE']
		# HOME = ''

		MODEL_PATH = HOME+f'/projects/{proj_id}/models/'
		if not os.path.isdir(MODEL_PATH):
			Path(MODEL_PATH).mkdir(parents=True)

		with open(MODEL_PATH+f'model_{item_id}.p', 'wb') as f:
			p.dump(model, f)

		files['X_file'].close()
		files['y_file'].close()

		return model
	
	def train_all_items(self, proj_id, retrain=True):

		item_ids = [int(x.split('_')[1]) for x in self.price_columns]
		processes = []
		results_ls = []

		with ThreadPoolExecutor(max_workers=5) as executor:

			for item_id in item_ids:
				processes.append(executor.submit(self.get_model,item_id,proj_id))

		for task in as_completed(processes):
			results_ls.append(task.result())

		# Check That All Models are Trained, Else Retrain
		HOME = os.environ['HOME_SITE']
		# HOME = ''
		proj_path = HOME+f'/projects/{proj_id}/'
		model_filenames = os.listdir(proj_path+'models')

		finished = 0
		while finished == 0:
			model_filenames = os.listdir(proj_path+'models')
			trained_models = [int(x.split('_')[1].split('.')[0]) for x in model_filenames]

			for item_id in item_ids:
				processes_incomplete = []
				results_ls_incomplete = []
				with ThreadPoolExecutor(max_workers=5) as executor:
					if item_id not in trained_models:
						# Retrain and Re-Save to Disk
						processes_incomplete.append(executor.submit(self.get_model,item_id,proj_id))
				for task in as_completed(processes_incomplete):
					results_ls_incomplete.append(task.result())

			model_filenames = os.listdir(proj_path+'models')
			trained_models = [int(x.split('_')[1].split('.')[0]) for x in model_filenames]

			if len(trained_models) == len(item_ids):
				finished = 1




		log.info(f'TRAINING COMPLETED FOR {len(item_ids)} ITEMS.')
		shutil.rmtree('temp/train')
