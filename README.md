# Image Classifier

A convolutional neural network trained on the CIFAR-10 dataset. It can classify images that fall within one of the CIFAR -10 categories with about 79% accuracy (and much better for some categories).

This program can test locally stored images. It also runs tests against the CIFAR-10 test image suite and can train new models (one is included in `resources` and used by default because training the model is resource-intensive and can take over an hour).

For more information about CIFAR-10, see:

[Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

[The CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## Installation

(NOTE: This program is theoretically compatible with any operating system that runs Python but is untested on Linux. Linux support should be considered experimental.)

- Install Python 3.12.5 [here](https://www.python.org/downloads/) (the most recent stable release at the time of writing) if not installed yet. (The program should be compatible with Python versions as old as 3.9, but this is untested.)
NOTE: On Windows, be sure to check the box marked “Add Python to PATH” in the installer, or the following instructions will not work.

- Download and unzip this program's source code.

- Using a terminal or command prompt, navigate into the source code directory (you should see a file called `requirements.txt`).

- Run the following command to install third-party dependencies, including Jupyter Notebook: `python -m pip install -r requirements.txt`

## Starting the Program

- To run the Jupyter Notebook (highly recommended):

  - Run the following command: `python -m notebook`

  - If the Jupyter Notebook interface doesn’t automatically open in your web browser, copy and paste one of the URLs that appear in the command prompt.

  - Within Jupyter Notebook’s file explorer, navigate to the source code directory and double-click `image_classification.ipynb` to open this program.

- To run the Python file:

  - Run the following command: `python ./image_classification.py`

## Usage

### Juypter Notebook

- First, select the code cell immediately below the `Class and Method Definitions` header and press `Run`. This cell loads the class instance that drives the notebook’s code into memory. NO OTHER CELLS WILL WORK UNTIL THIS CELL IS RUN!
NOTE: This cell may take a while to run for the first time because it must download the CIFAR-10 dataset.

- Take a look at the included instructions and explanations. They will guide you as you explore the program’s features.

- TESTS: The Jupyter Notebook allows you to experiment with three types of tests. They are briefly explained here; see the Notebook file for examples and detailed usage instructions.

  - Local images test: Classify a locally stored image. For convenience, a sample public-domain photo of an airplane is included in `resources/example-airplane-photo.jpg`.

  - CIFAR-10 test images accuracy test: Test this program’s accuracy against the entire CIFAR-10 test image suite. The current model shows an accuracy of about 79%.

  - CIFAR-10 single category accuracy test: Test this program’s accuracy against 100 images randomly selected from a single category of your choice. You may choose any of the ten categories in CIFAR-10. After the test, the results are shown, and a collated image of all correctly and incorrectly categorized images is displayed.

- DATA VISUALIZATIONS: Four different visualizations are available, each based on the model’s input data or output predictions (for detailed information about each image, see the relevant sections in the Notebook file).

  - Bar chart: This visualization shows the model’s percent accuracy per category.

  - Confusion matrix: This visualization shows the model’s miss and success count for each category.

  - Scatterplot (test images): This visualization shows the average proportion of red, green, and blue pixel weights per category in the CIFAR-10 test image suite.

  - Scatterplot (training images): This visualization shows the average proportion of red, green, and blue pixel weights per category in the CIFAR-10 training image suite.

- REGENERATE THE MODEL: Re-train the provided model using the CIFAR-10 training set. Depending on your machine, this could take up to several hours. Once generated, the new model will overwrite the model currently saved in `resources/image-classifier.keras` unless you rename that file.

### Python file

- In the menu, type `1`, `2`, `3`, `4`, or `5` and press `enter` to test an option, or type `6` and `enter` to quit.

- OPTION 1: Test a local image. Type the absolute path (or the relative path from the `.py` file) to an image, and this program will predict its classification.

        Example: 
        
        resources/example-airplane-photo.jpg

        Output: 
        
        Most likely type: airplane

- OPTION 2: Test against the CIFAR-10 test suite, all categories. This option tests this program against the CIFAR-10 test suite and returns its accuracy score.

- OPTION 3: Test against the CIFAR-10 test suite, single category. You will be prompted to select a category. After doing so, this program will return its best guess for 100 random images in the category. Press enter to display a collage of the photos, with red “x”s over images the model guessed incorrectly.

- OPTION 4: Show data visualizations. For detailed information about each image, see the portion about “Data Visualizations” in the Jupyter Notebook usage section.

- OPTION 5: Regenerate model. For details, see “Regenerate the Model” in the Jupyter Notebook usage section.

- OPTION 6: Exit the program.
