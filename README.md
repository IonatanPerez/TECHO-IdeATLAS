# About

On this project we will implement some algorithms to discover georeferenced information using data from the open Buildings project of google: https://sites.research.google/gr/open-buildings/ . We want to measure the orientation and factor of coverage of the buildings in the area of BaAs.

## File structure

The files [`1-Exploration.ipynb`](1-Exploration.ipynb) and [`2-Processing.ipynb`](2-Processing.ipynb) are used to explore the data and understand the problem. On those files there is partial codes pieces, some test and the preliminary versions of the final code. Those files are share to expose the process of thinking the project.

The file [`3-DataPreparation.ipynb`](3-DataPreparation.ipynb) is a small file used to prepare small datasets from the big source file that has a huge region and is very heavy to be shared on the project.

On [`Project Description.md`](Project%20Description.md) there is a detailed explanations of the logic under the pipeline of the project.

The [`Pipeline.ipynb`](Pipeline.ipynb) file is a notebook that contains the main steps of the project with a minor presentation of internal steps. This file is used to check that everything is working ok on the development process and produce the required outputs.

[`settings.py`](settings.py) has the common settings and hardcoded values used on the project.

[`tools.py`](tools.py) is the main file with the functions used to process the data.

On [`requirements.txt`](requirements.txt) there are the libraries used on the project.

The `output` folder contains the outputs of the project.