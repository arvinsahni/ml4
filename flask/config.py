WTF_CSRF_ENABLED=True
SECRET_KEY='ml4all'

OPENID_PROVIDERS = [ 
	{'name':'Google','url':'https://www.google.com/accounts/o8/id'},
	{'name':'Yahoo','url':'https://me.yahoo.com'},
	{'name':'Flickr','url':'https://www.flickr.com/<username>'},
	{'name':'AOL','url':'https://openid.aol.com/<username>'},
	{'name':'MyOpenID','url':'https://www.myopenid.com'}]

import os
basedir =os.path.abspath(os.path.dirname(__file__))

SQLALCHEMY_DATABASE_URI='sqlite:///'+os.path.join(basedir,'app.db')
SQLALCHEMY_MIGRATE_REPO=os.path.join(basedir,'db_repository')
SQLALCHEMY_TRACK_MODIFICATIONS=False
MAX_CONTENT_LENGTH = 32 * 1024 * 1024
UPLOAD_FOLDER='uploads/'
ALLOWED_EXTENSIONS=set(['txt','csv'])
