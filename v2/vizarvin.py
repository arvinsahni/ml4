from __future__ import division

from flask import render_template, request, Response, jsonify,redirect,url_for,flash
from werkzeug.utils import secure_filename

from app import app

from pandas.util import hash_pandas_object


import json
import psycopg2
import psycopg2.extras
import os
import pandas as pd
import hashlib
import datetime
from datetime import date
import numpy as np

TRAINING_DATA={}
TESTING_DATA={}


ALLOWED_EXTENSIONS=set(['txt','csv'])
SECRET_KEY='ml4all'
app.secret_key='ml4all'

@app.route('/index')
def index():
	return render_template('home.html')


@app.route('/viz')
def viz():
	return render_template('viz.html')


def to_csv(d, fields):
	d.insert(0, fields)
	return Response('\n'.join([",".join(map(str, e)) for e in d]), mimetype='text/csv')

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def datset():
   return render_template('home.html')

@app.route('/dataset',methods=['POST'])
def upload_file():
	train_file_name = 'train'
	test_file_name ='test'
	error=None
	if request.method == 'POST':

		# check if the post request has the file part
		if train_file_name not in request.files or test_file_name not in request.files:
			#flash('No file part')
			error='Kindly upload both training and testing files'
			#print("helllllo")
			#print(request.url)
			flash("load files")
			#return redirect(request.url)
			return render_template('home.html',error=error)


		file = request.files[train_file_name]

		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':

			print("hiiio")
			print(request.url)
			error='Kindly upload both training and testing files'

			flash('No selected files')
			return redirect(request.url)
			#return render_template('home.html',error=error)

		if file and allowed_file(file.filename):
			flash("training file uplaoded")
			filename = secure_filename(file.filename)
			print(os.path.abspath(os.path.join('app/','uploads/')))
			#file.save(os.path.abspath(os.path.join('app/',app.config['UPLOAD_FOLDER'], filename)))
			file.save(os.path.abspath(os.path.join('app/','uploads/', filename)))
			print("done")
			## convert file to pandas dataframe
			#df_train=pd.read_csv(os.path.join('app/',app.config['UPLOAD_FOLDER'], filename))
			df_train=pd.read_csv(os.path.join('app/','uploads/', filename))

			print("df_train1",df_train.head(5))

			## hash the pd , change to binary --> get fom Jason
			temp_hash=hash_pandas_object(df_train)
			hash_train = hashlib.sha256(str(temp_hash).encode('utf-8','ignore')).hexdigest()
			print("hash train1",hash_train)

			## update dict ---> key:hash ,value: dataframe
			#TRAINING_DATA[hash_train]=df_train

		## For the test file
		file = request.files[test_file_name]

			# if user does not select file, browser also
			# submit a empty part without filename
		if file.filename == '':
			print(request_url)
			flash('No selected files')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			#file.save(os.path.abspath(os.path.join('app/',app.config['UPLOAD_FOLDER'], filename)))
			file.save(os.path.abspath(os.path.join('app/','uploads/', filename)))

			## convert file to pandas dataframe
			#df_test=pd.read_csv(os.path.join('app/',app.config['UPLOAD_FOLDER'], filename))
			df_test=pd.read_csv(os.path.join('app/','uploads/', filename))
			print("df test1",df_test.head(5))

			## hash the pd , change to binary --> get fom Jason
			temp_hash=hash_pandas_object(df_test)
			hash_test = hashlib.sha256(str(temp_hash).encode('utf-8','ignore')).hexdigest()
			print("test1",hash_test)

			## update dict ---> key:hash ,value: dataframe

			if df_train.shape[1]==(df_test.shape[1]-1):
				temp=hash_test
				hash_test=hash_train
				hash_train=temp
				temp_df=df_test
				df_test=df_train
				df_train=temp_df

			TESTING_DATA[hash_test]=df_test
			TRAINING_DATA[hash_train]=df_train
			print("hash train2",hash_train)
			print("hash test2",hash_test)
			print("df train2",df_train)
			print("df_test2",df_test)
			flash("Uploaded files all training")
			#return redirect('home.html')
			#return jsonify({"hash":hash})
			#return redirect(request.url)
			return redirect(url_for('datset'))

## may look to add another app.route for test data hash but later

#@app.route('/dataset_test',methods=['POST'])
#def upload_testfile():


#        file_name = 'test[]'
#
#        if request.method == 'POST':
#
#                # check if the post request has the file part
#                if file_name not in request.files:
#                        print(request.files)
#                        flash('No file part')
#                        return redirect(request.url)
#
#                file = request.files[file_name]
#
#                print (file.filename)
#                # if user does not select file, browser also
#                # submit a empty part without filename
#                if file.filename == '':
#
#                        flash('No selected files')
#                        return redirect(request.url)
#                if file and allowed_file(file.filename):
#
#                        filename = secure_filename(file.filename)
#
#                        print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#                        print(os.getcwd())
#                        file.save(os.path.join('app/',app.config['UPLOAD_FOLDER'], filename))
#
#                        ## convert file to pandas dataframe
#
#                        df_test=pd.read_csv(os.path.join('app/',app.config['UPLOAD_FOLDER'], filename))
#                        print(df_test.head(5))

#                        ## hash the pd , change to binary --> get fom Jason
#                        temp_hash_test=hash_pandas_object(df_test)

#                        print(temp_hash_test)
#                        testing_data_hash = hashlib.sha256(str(temp_hash_test).encode('utf-8','ignore')).hexdigest()
#                        print(testing_data_hash)
#                        ## update dict ---> key:hash ,value: dataframe
#                        TESTING_DATA[temp_hash_test]=df_test

#                        return jsonify({"test_data_hash":testing_data_hash})


BASIC_STATS = {}
##replace with actual function
def jacky_function(df):
	return date.today(),1,len(list(df))

@app.route('/basic-stats/<hash>',methods=['GET'])
def basic_stat(hash):
	## step 1 if hash in BASIC_STATS return jsonify(BASIC_STATS[hash])
	## else step 2
	## pull in training data
        ## compute basic stats(basically call Jacky's function)  add results to dictionary BASIC_STATS, return jsonify(BASIC_STATS[hash])
        ## which is basically {"metadata": {"date": <ISO Format>, "version": <int>}, "data": {<data collection 1>: {}, <data collection 2>: {}, ...}}
	print(TRAINING_DATA)

	if hash in BASIC_STATS:
		return jsonify({BASIC_STATS[hash]})
		# error can be sent the same way jsonify(BASIC_STATS[error])
	else:
		#for key,value in TRAINING_DATA.items():
			#print (key,value)
		train_df=TRAINING_DATA[hash_train]
		date_fn,version_fn,stats=jacky_function(train_df)
		BASIC_STATS[hash_train]=stats
		return jsonify({"metadata":{"date":str(date_fn),"version":version_fn},"data":stats})


## Prediction stats work the same way as basic stats except i need to call Jason's function instead of Jacky's function
## this would need a MODELS dictionary - key is the hash value, value is the model we train
## input to a function that Jason will write ---> model ( from the MODELS dictionary)
## output would be {"metadata": {"date": <ISO Format>, "version": <int>}, "data": {"technical_scores": [{"name": "AUC", "value": .867}, {"name": "Accuracy", "value": "79%"}], <data collection 2>: {}, ...}}
# ( inform JAcky of the structure --how its returned)

MODELS={}
##replace with actual function
sample={}
temp={}
temp
sample["technical_scores"]=[]
def jason_function(df):
        return date.today(),100,len(list(df))

@app.route('/prediction-stats/<hash>',methods=['GET'])
def prediction_stat(hash):
	print(TRAINING_DATA)
	if hash in MODELS:
        	return jsonify({MODELS[hash]})
	else:
		train_df=TRAINING_DATA[hash]
		date_fn,version_fn,pred_stats=jason_function(train_df)
		MODELS[hash]=pred_stats
		return jsonify({"metadata":{"date":str(date_fn),"version":version_fn},"data":pred_stats})



## test data prediction

#replace by actual code
def jason_model_creation(hash):
	return 100
def jason_prediction(model_saved,hash,testing_data_hash):
	return 200

## this one should return a df
def jason_add_pred_to_test(pred,testing_data_hash,hash):
	return pd.DataFrame(np.random.randn(10, 5))

from flask import send_from_directory
MODELS_SAVED={}
## the below is for checking stuff only
#TESTING_DATA["e0d47420dd0157af6af54d64b14f348f1fada3c050a73cd50fad2716a38fc2b2"]=1234
@app.route('/predict/<hash>/<testing_data_hash>',methods=['GET'])
def prediction_test(hash,testing_data_hash):
	print("step1")
	print(TRAINING_DATA)
	for key, value in TRAINING_DATA.items() :
		print (key, value)

	if hash not in MODELS_SAVED:
		train_df=TRAINING_DATA[hash]
		test_df=TESTING_DATA[testing_data_hash]
		temp=jason_model_creation(hash)
		## replace above based on actual model
		MODELS_SAVED[hash]=temp
		print("step 2")
	pred=jason_prediction(MODELS_SAVED[hash],hash,testing_data_hash)
	pred_df=jason_add_pred_to_test(pred,testing_data_hash,hash)
	print("step 3")
	pred_filename="abcd.csv"  # need to have some component of date,version etc
	pred_df.to_csv(pred_filename)
	print(os.getcwd())
	#pred_filename = secure_filename(pred_filename)
	#file.save(os.path.join('app/',app.config['UPLOAD_FOLDER'], pred_filename))
	#return send_from_directory(app.config['UPLOAD_FOLDER'],pred_filename)
	print("step 4")
	return send_from_directory(os.getcwd(),pred_filename)

## may add 2 more app.routes
