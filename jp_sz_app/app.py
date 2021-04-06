from flask import Flask, request, render_template, flash, redirect, url_for, session
import pickle
import gc
import numpy as np

model = pickle.load( open( "nb_vect.p", "rb" ) )

app = Flask(__name__)
app.secret_key = "super secret key"


@app.route('/', methods=["GET","POST"])
def homepage():

    if request.method == "POST":
        pred_arr = model.predict_proba([request.form['text_form']])

        pred_class = np.argmax(pred_arr[0])

        if pred_class == 0:
            score = pred_arr[0][0]
            score = int(score*100)
            return redirect(url_for('zizek',score=score))
        elif pred_class == 1:
            score = pred_arr[0][1]
            score = int(score*100)

            return redirect(url_for('jbp',score=score))

    return render_template("index.html")

@app.route('/zizek', methods=["GET","POST"])
def zizek():

    score = request.args.get('score')
    if score:
        return render_template("zizek.html",score=score)

    return render_template("index.html")

@app.route('/jbp', methods=["GET","POST"])
def jbp():

    score = request.args.get('score')
    if score:
        return render_template("jbp.html",score=score)

    return render_template("index.html")


if __name__ == "__main__":
	app.run()
