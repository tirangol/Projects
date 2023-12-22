"""Climate Net - GUI Part"""

from flask import Flask, render_template, request, jsonify
from preprocessing import get_prediction, preprocess, unflatten_data, get_koppen, get_trewartha
from model import climate_net
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()

        elevation = np.array(data['elevation'])
        land = (elevation > float(data['water_threshold'])).astype(bool)

        preprocess(elevation, land)
        prediction = get_prediction(climate_net)

        temperature = unflatten_data(prediction[:, :12])
        precipitation = unflatten_data(prediction[:, 12:])
        koppen = get_koppen(temperature, precipitation, land)
        trewartha = get_trewartha(temperature, precipitation, elevation, land)

        temperature[:, ~land] = 0
        precipitation[:, ~land] = 0
        koppen[~land] = 0
        trewartha[~land] = 0

        return jsonify(result=[temperature.tolist(),
                               precipitation.tolist(),
                               koppen.tolist(),
                               trewartha.tolist(),
                               land.tolist(),
                               [float(np.nanmin(temperature)), float(np.nanmax(temperature))],
                               [float(np.nanmin(precipitation)), float(np.nanmax(precipitation))]])
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
