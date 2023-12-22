# Climate Net

A work-in-progress app that uses a neural network to predict the climate of Earth-like planets, intended as a fast & high-resolution worldbuilding tool for generating realistic climates. Earth-like in this context means matching modern-era Earth in every property except for the elevation map.

It supports inputs in the form of images (.png, .jpg, .tif), text (eg. `[[1, 2, 3], [4, 5, 6]]`), or randomly generated terrain, and contains a editor that can perform limited elevation-editing operations.

Outputs come in the form of monthly temperature, monthly precipitation, and climate classification maps (currently supports Koppen and Trewartha).

Python was used for the machine learning component - Pillow, Scipy and Numpy for the data processing, Pytorch for the neural network and training process, and Flask for the web part. HTML, CSS, and Javascript were used to build the website as well as the editor.

<p align="center">
<img src="gifs/editing.gif" width="1000px">
</p>

<p align="center">
<img src="gifs/results.gif" width="1000px">
</p>

Currently, only the minimum functionality has been implemented. There are a lot of things I'm working on, such as:

- Optimizing the preprocessing speed (which is far too slow)
- More realistic random terrain generation
- More options for elevation colour scale
- Global elevation offsetting (rather than just changing the water level)
- GIF, PNG, and text exporting of results
- More colour schemes for results
- A colour bar for temperature/precipitation and legend for climate classifications
- Add Thornthwaite climate classification system
- Optional logarithmic scale for precipitation
- Generating temperature-precipitation plots of individual pixels
- Statistics about the land - elevation distribution, percentage of land/water cover
- For a pixel, finding similar-climate cities in the real-world

To get started, run `gui.py` and go to the web address specified by the console after it starts running.

If you just want to use the Python part, you'll need two `180 x 360` numpy arrays as your inputs - one is your elevation map (a float array) and the other is the land map (a boolean area where 0/1 = water/land). Run the code in `model.py`, then the code in `preprocessing.py`, then the following lines:

```
preprocess(elevation, land)                   # where elevation, land are your two numpy arrays
prediction = get_prediction(climate_net)

temperature = unflatten_data(prediction[:, :12])
precipitation = unflatten_data(prediction[:, 12:])
koppen = get_koppen(temperature, precipitation, land)
trewartha = get_trewartha(temperature, precipitation, elevation, land)
```

Of the last four variables, the first two are numpy arrays of size `12 x 180 x 360`, while the last two are of size `180 x 360`. Use them however you wish.
