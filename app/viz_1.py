from __future__ import division

from flask import render_template, request, Response, jsonify, send_from_directory
from app import app

import json
import psycopg2
import os
import sys
import psycopg2.extras
import pandas as pd

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from learn import forall as fa
from learn import utils

@app.route('/index')
def index():
    return render_template('home.html')


@app.route('/viz')
def viz():
    return render_template('viz.html')


def to_csv(d, fields):
    d.insert(0, fields)
    return Response('\n'.join([",".join(map(str, e)) for e in d]), mimetype='text/csv')


@app.route('/hist_data', methods=['GET', 'POST'])
def hist_data():
    website = request.args.get('website')
    person = request.args.get('person')
    db = psycopg2.connect(host='ec2-54-208-219-223.compute-1.amazonaws.com',
                          database='election',
                          user='elections',
                          password='election2016')
    curs = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    DEC2FLOAT = psycopg2.extensions.new_type(
        psycopg2.extensions.DECIMAL.values,
        'DEC2FLOAT',
        lambda value, curs: float(value) if value is not None else None)
    psycopg2.extensions.register_type(DEC2FLOAT)

    if website:
        sql = """select a.bin, sum(coalesce(count,0)) from histogram_bins a
                left join (select * from data_binned where website = '%s' and person = '%s') b on a.bin = b.bin
                group by 1 order by 1""" % (website, person)
    else:
        sql = """select a.bin, sum(coalesce(count,0)) from histogram_bins a
                left join (select * from data_binned where person = '%s') b on a.bin = b.bin
                group by 1 order by 1""" % person
    print(sql)
    curs.execute(sql)
    d = curs.fetchall()
    print(d)
    fields = ('bin', 'sum')
    return jsonify(data=d)


@app.route('/dataset', methods=['POST'])
def dataset():
    # print(request.get_data())
    print(request.files)

    dtrain = request.files['train']
    dtest = request.files['test']

    #Save input data files in input folder
    dtrain.save("input/" + dtrain.filename)
    dtest.save("input/" + dtest.filename)

    df_train = pd.read_csv("input/" + dtrain.filename)
    # print(df_train.head())

    df_test = pd.read_csv("input/" + dtest.filename)
    # print(df_test.head())

    #From Jason's ML module
    X, y = utils.X_y_split(X_train=df_train, X_test=df_test)
    model = fa.All()
    model.fit(X, y)

    #Append prediction column to test set
    predictions = model.predict(df_test)
    df_test['prediction'] = predictions

    #Save prediction in output folder
    print(df_test.head())
    df_test.to_csv("output/" + "prediction.csv", index=False)

    print("%s: %.3f (%s)" % ("Jacky's data:", model.score, model.score_type))
    return '{ "fake_json":100}', 200


@app.route('/download')
def download(filename=None):
    # uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
    return send_from_directory(directory=os.path.abspath(os.path.join('../flask/output')), filename="prediction.csv")
    # return '{ "fake_json":100}'
