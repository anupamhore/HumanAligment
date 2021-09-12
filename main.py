from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from trainingValidation import train_validation
from trainingModel import TrainModel


os.putenv('LANG','en_US.UTF-8')
os.putenv('LC_ALL','en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():

    try:
        if request.json['filepath'] is not None:
            path = request.json['filepath']

            # clean, merge the csv files
            # train_val_obj = train_validation()
            # train_val_obj.startValidation()

            # start training
            trainingModelObj = TrainModel()
            trainingModelObj.modelTraining()



    except ValueError:
        return Response("Error Occurred! %s" % ValueError)

    except KeyError:
        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:
        return Response("Error Occurred! %s" % e)

    return Response("Training Successfull!!")




if __name__ == '__main__':
    app.run(debug=True)


