# Neuromorphic research notes

In this repository part of the work done for the neuromorphic part of the [CROSSBRAIN](https://crossbrain.eu/) project.
What is contained here is mostly partial, collecting mostly notes about work in progress.

## Contents

- [Contents](#contents)
- [Project Structure](#project-structure)
- [Useful links](#useful-links)
- [Software-Hardware convertion](#software-hardware-convertion)
- [Data](#data)
  - [Experimental Signals](#experimental-signals)
  - [Sources](#sources)
  - [Datasets Candidates](#datasets-candidates)
- [Frameworks](#frameworks)

## Project Structure

- [Experiments](./experiments/). Experiments folders. Each one contains a *flows* folder.
- [Tutorial Apps](./streamlit_apps/). Streamlit Apps to get a better feeling about neuron models, using snnTorch.
- [stdp](./stdp/). A library dedicated to the experiments on STDP running here. It also contains a module with estimators classes.
- [experimentkit_in](./experimentkit_in/). A library containing tools for experiments.
- *tests/* contains test functions for the libraries in this projects.

## Useful links

- [Google Drive Folder](https://drive.google.com/drive/folders/1mg8L234w0UKHV8RTb_CtaCxzJC0KfjkW)
- [medphy-unitov/CROSSBRAIN_signals](https://github.com/medphy-unitov/CROSSBRAIN_signals)

## Software-Hardware convertion

The Crossbrain project is aimed at making physical devices, reasoning that it is necessary to have a clear policy of converting software work into hardware work.
Defining such a policy depends on several variables, including the framing of the problem and the tools at hand, as well as contextual design decisions.

See [sw-to-hw](./doc/sw-to-hw.md) for more details.

## Data

### Experimental Signals

- [Data256Channels2021, Description](https://istitutoitalianotecnologia-my.sharepoint.com/:w:/r/personal/matteo_vincenzi_iit_it/_layouts/15/Doc.aspx?sourcedoc=%7B8C3AB520-D1CF-4DB4-9BB6-56C596B71C51%7D&file=README.docx&action=default&mobileredirect=true), from IIT;
- Kilosort spike sorted data, from IIT.

### Sources

- https://physionet.org/
- https://openneuro.org/
- https://arxiv.org/pdf/2208.08860.pdf

### Datasets Candidates

See [datasets](./docs/datasets.md).

Other references  

1. https://github.com/meagmohit/EEG-Datasets
2. [BCI Competition IV: Download area, Berlin Brain-Computer Interface](https://www.bbci.de/competition/iv/download/index.html?agree=yes&submit=Submit), Data set 1, 100Hz data. 
3. [TIBREWAL, Navneet; LEEUWIS, Nikki; ALIMARDANI, Maryam. Classification of motor imagery EEG using deep learning increases performance in inefficient BCI users. Plos one, 2022, 17.7: e0268880.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0268880)

## Frameworks

1. [SpikeSort](https://spike-sort.readthedocs.io/en/latest/intro.html)
2. [MEG/EEG analysis with MNE-Python](https://mne.tools/dev/auto_tutorials/intro/10_overview.html) 
3. [tslearn](https://tslearn.readthedocs.io/en/stable/index.html)
