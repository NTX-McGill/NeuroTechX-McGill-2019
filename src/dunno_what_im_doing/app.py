from flask import Flask, render_template, request

# create flask app
app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        if request.form['submit_button'] == 'left':
            with open("../../offline/training_software/settings.txt", 'w') as f:
                f.write('left')
                f.close()
        elif request.form['submit_button'] == 'right':
            with open("../../offline/training_software/settings.txt", 'w') as f:
                f.write('right')
                f.close()
        elif request.form['submit_button'] == 'rest':
<<<<<<< HEAD
            with open("../../offline/training_software/settings.txt", 'w') as f:
                f.write('rest')
                f.close()
        elif request.form['submit_button'] == 'wait':
=======
>>>>>>> 49ef48035381c48124343e94981ace4560a274a6
            with open("../../offline/training_software/settings.txt", 'w') as f:
                f.write('rest')
                f.close()
    return render_template("index.html")
