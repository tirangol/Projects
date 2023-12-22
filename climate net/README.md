# Climate Net

A work-in-progress app that uses a neural network to predict the climate of Earth-like planets, intended as a fast & high-resolution worldbuilding tool for generating realistic climates. Earth-like in this context means matching modern-era Earth in every property except for the elevation map.

It supports inputs in the form of images (.png, .jpg, .tif), text (eg. `[[1, 2, 3], [4, 5, 6]]`), or randomly generated terrain, and contains a editor that can perform limited elevation-editing operations.

Outputs come in the form of monthly temperature, monthly precipitation, and climate classification maps (currently supports Koppen and Trewartha).

Python was used for the machine learning component - Pillow, Scipy and Numpy for the data processing, Pytorch for the neural network and training process, and Flask for the web part. HTML, CSS, and Javascript were used to build the website as well as the editor.

<p align="center">
<img src="gifs/editing.gif" width="500px">
</p>

<p align="center">
<img src="gifs/results.gif" width="500px">
</p>

Currently, only the base functionality has been implemented; there a lot of things to add, such as:

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
