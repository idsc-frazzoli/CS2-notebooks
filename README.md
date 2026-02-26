# [CS2-2024-notebooks]

This GitHub repository contains the Juypter notebooks for the [Control Systems II](https://idsc.ethz.ch/education/lectures/control-systems-ii.html) course taught by Prof. Emilio Frazzoli at ETH Zürich, spring semester 2024.

## Getting Started with the Notebooks

There are two ways to use the notebooks. Either you download them locally on the computer where you have Python and Jupyter installed, or you simply open them in Google Colab and start coding directly in your browser. We recommend that you use [VS Code](https://code.visualstudio.com/) for your editor. If you need help setting up the notebooks locally, see the installation guide below.


## Installation Guide: Jupyter Notebooks Setup

### Prerequisites
Before proceeding, ensure you have Git installed on your system to clone the repository. If not, you can download it from [Git's official website](https://git-scm.com/downloads).

### Cloning the Repository from GitHub

To clone the repository, open a terminal or command prompt and run the following command:

    git clone https://github.com/idsc-frazzoli/CS2-2024-notebooks.git


### Manually Downloading the Notebooks

Alternatively, you can manually download the notebooks:
1. Navigate to the [GitHub repository](https://github.com/idsc-frazzoli/CS2-2024-notebooks) in your web browser.
2. Find the `Code` button and click it, then select `Download ZIP`.
3. Extract the ZIP file to your desired directory.

### Setting Up the Environment

#### Windows

1. Install Python from the [official Python website](https://www.python.org/downloads/). You need Python version 3.9 or higher.
2. Open Command Prompt and navigate to the project directory.
3. Create a virtual environment:

        python -m venv venv

4. Activate the virtual environment:

        .\venv\Scripts\activate

5. Install Jupyter:

        pip install jupyter


#### Linux and Mac

1. Install Python by downloading it from the [official Python website](https://www.python.org/downloads/) or using a package manager (e.g., `apt` for Ubuntu, `brew` for macOS). Ensure it is version 3.9 or higher.
2. Open a terminal and navigate to the project directory.
3. Create a virtual environment:

        python3 -m venv venv

4. Activate the virtual environment:
- On Linux:
  ```
  source venv/bin/activate
  ```
- On Mac:
  ```
  source venv/bin/activate
  ```
5. Install Jupyter:

        pip install jupyter



### Setting Up Jupyter Kernel

After activating your virtual environment and installing Jupyter, set up the Jupyter kernel with the following command:

        python -m ipykernel install --user --name=venv


Replace `venv` with the name of your virtual environment if you used a different name.

### Starting Jupyter Notebook

To start Jupyter Notebook, run:

        jupyter notebook


This will open Jupyter Notebook in your default web browser, allowing you to open and run the notebooks you've cloned or downloaded.

### Additional Setup for VSCode

If you are using VSCode, follow these additional steps to integrate your setup:
1. Install the Python and Jupyter extensions for VSCode.
2. In VSCode, open the command palette (Ctrl+Shift+P) and select `Python: Select Interpreter`. Choose the virtual environment you created.
3. Open a notebook file (`.ipynb`), and you should be able to run the cells using the Python kernel you've set up.


## Installation Guide: Google Colab
All notebooks are available on this [CS2 Google Drive folder](https://drive.google.com/drive/u/1/folders/1TrbJvXfJnbALlsuXLY7iIzd8rI-fRAU1
). You need to open the files as a new copy with [Google Colab](https://colab.research.google.com/)
The kernel runs automatically from the browser and no additional installations must be made. 

## Usage
Each notebook consists of controls case studies and coding exercises. All dependencies and background functions are handled by the [cs2solutions](https://github.com/idsc-frazzoli/cs2solutions) package. Please ensure that this is installed at the start of the notebook using ```pip install cs2solutions```.

This package contains useful functions, unit tests, and solutions. More information is found on it's official repository: [cs2solutions](https://github.com/idsc-frazzoli/cs2solutions)
