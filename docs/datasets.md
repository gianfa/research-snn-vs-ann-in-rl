# Datasets Candidates

\[ from https://github.com/medphy-unitov/CROSSBRAIN_signals/edit/develop/doc/datasets.md \]

- [Tested](#tested)
  - [1 ECoG in epilepsy from RESPect (Registry for Epilepsy Surgery Patients), a dataset recorded at the University Medical Center of Utrecht](#1-ecog-in-epilepsy-from-respect-registry-for-epilepsy-surgery-patients-a-dataset-recorded-at-the-university-medical-center-of-utrecht)
  - [2 Local field potential activity dynamics in response to deep brain stimulation of the subthalamic nucleus in Parkinson's disease](#2-local-field-potential-activity-dynamics-in-response-to-deep-brain-stimulation-of-the-subthalamic-nucleus-in-parkinsons-disease)
  - [3 Dataset of cortical activity recorded with high spatial resolution from anesthetized rats](#3-dataset-of-cortical-activity-recorded-with-high-spatial-resolution-from-anesthetized-rats)
  - [4 Multimodal in vivo recording using transparent graphene microelectrodes illuminates spatiotemporal seizure dynamics at the microscale](#4-multimodal-in-vivo-recording-using-transparent-graphene-microelectrodes-illuminates-spatiotemporal-seizure-dynamics-at-the-microscale)
- [To be tested](#to-be-tested)
  - [1](#1)
  - [2 MNE FreeSurfer Average](#2-mne-freesurfer-average)
- [Other LFPs can be found here](#other-lfps-can-be-found-here)
- [Possible interest](#possible-interest)
  - [Early Detection of Human Epileptic Seizures Based on Intracortical Microelectrode Array Signals](#early-detection-of-human-epileptic-seizures-based-on-intracortical-microelectrode-array-signals)
  - [EEG datasets for seizure detection and prediction— A review](#eeg-datasets-for-seizure-detection-and-prediction-a-review)
  - [A dataset of neonatal EEG recordings with seizure annotations | Scientific Data](#a-dataset-of-neonatal-eeg-recordings-with-seizure-annotations--scientific-data)
  - [Detection of spontaneous seizures in EEGs in multiple experimental mouse models of epilepsy](#detection-of-spontaneous-seizures-in-eegs-in-multiple-experimental-mouse-models-of-epilepsy)

## Tested

### 1 ECoG in epilepsy from RESPect (Registry for Epilepsy Surgery Patients), a dataset recorded at the University Medical Center of Utrecht
paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9440951/  
data: https://openneuro.org/datasets/ds003844/versions/1.0.3  
ECoG in epilepsy from RESPect (Registry for Epilepsy Surgery Patients), a dataset recorded at the University Medical Center of Utrecht

LFP
requirements: mne, datalab (possibly)

### 2 Local field potential activity dynamics in response to deep brain stimulation of the subthalamic nucleus in Parkinson's disease
recordings in PD in three cases (ON meds, OFF meds, during DBS)

paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7115855/  
data: https://data.mrc.ox.ac.uk/stn-lfp-on-off-and-dbs  

### 3 Dataset of cortical activity recorded with high spatial resolution from anesthetized rats
High-resolution on rats
paper: https://www.nature.com/articles/s41597-021-00970-3#Sec27  
data: https://gin.g-node.org/UlbertLab/High_Resolution_Cortical_Spikes  

requirements: g-node (gin), spikeinterface

### 4 Multimodal in vivo recording using transparent graphene microelectrodes illuminates spatiotemporal seizure dynamics at the microscale
Induced seizure on mice
paper: https://www.nature.com/articles/s42003-021-01670-9#data-availability  
data: https://figshare.com/articles/dataset/Multimodal_in_vivo_recording_of_induced_seizures_Calcium_imaging_and_ECoG_using_transparent_graphene_microelectrode/13007840  

## To be tested 

### 1
Local and distant cortical responses to single pulse intracranial stimulation in the human brain are differentially modulated by specific stimulation parameters  
paper: https://www.sciencedirect.com/science/article/pii/S1935861X22000456   
data: https://dabi.loni.usc.edu/dsi/W4SNQ7HR49RL   
large dataset not tried ()

### 2 MNE FreeSurfer Average
MNE Intracranial EEG dataset.  
data: https://github.com/mne-tools/mne-misc-data/tree/main/seeg  
example: https://mne.tools/mne-gui-addons/auto_examples/ieeg_locate.html#tut-ieeg-localize


## Other LFPs can be found here 
https://data.mrc.ox.ac.uk/mrcbndu/data-sets/search?page=1

## Possible interest
### Early Detection of Human Epileptic Seizures Based on Intracortical Microelectrode Array Signals
https://ieeexplore.ieee.org/document/8732415/media#media

### EEG datasets for seizure detection and prediction— A review  
\[Suggested by M.Giugliano\]  
https://onlinelibrary.wiley.com/doi/10.1002/epi4.12704e
EEG datasets for seizure detection and prediction— A review


### A dataset of neonatal EEG recordings with seizure annotations | Scientific Data
\[Suggested by M.Giugliano\]  
Human data.
https://www.nature.com/articles/sdata201939


### Detection of spontaneous seizures in EEGs in multiple experimental mouse models of epilepsy
\[Suggested by M.Giugliano\]  
Mouse models.
https://iopscience.iop.org/article/10.1088/1741-2552/ac2ca0/pdf  
https://lifescience.ucd.ie/Epi-AI/  
>...two mouse models of kainic acid-induced epilepsy (Models I and III), a genetic model of Dravet syndrome (Model II) and a pilocarpine mouse model of epilepsy (Model IV).

  

<!-- [This is a comment: it's not visible after publication]
    The link below is just to go back :)
 -->  
  
Back to [README](./../README.md)
