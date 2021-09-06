import classifier
from flask import Flask, request, render_template

#instantiate flask method
app = Flask(__name__, template_folder='templates')

#connect to the home landing page where users can enter data
@app.route('/home')
def home():
    return render_template('home.html')


#connect the python webserver and outpage page which will show the predicted value
@app.route('/classify', methods=['GET'])
def classify_type():

    try:
        sepal_len = request.args.get('slen') # Get parameters for sepal length
        sepal_wid = request.args.get('swid') # Get parameters for sepal width
        petal_len = request.args.get('plen') # Get parameters for petal length
        petal_wid = request.args.get('pwid') # Get parameters for petal width

        variety = classifier.predict(sepal_len, sepal_wid, petal_len, petal_wid)  

        return render_template('output.html', variety=variety)
    except:
        return 'Error'

if(__name__=='__main__'):
    app.run(debug=True)
