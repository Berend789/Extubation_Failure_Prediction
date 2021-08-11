# Extubation_Failure_Prediction
This is a complete preprocessing, model training, and figure generation repo for "THE USE OF MACHINE LEARNING ON PICU DATA FOR THE PREDICTION OF EXTUBATION FAILURE AFTER SURGERY IN PEDIATRIC PATIENTS WITH CONGENITAL HEART DISEASE "

**Build up** <br />
All files are build up similar. <br /> All files with the CHD extension were specified for congenital heart disease cohort, without the extension were specified for the bronchiolitis cohort.  In all files the output_folder is the main folder for results

CHD_dat_prep: Chunkwise preparation of raw data <br />
LR_Build_CHD: Functions to prepare data for analysis <br />
LR_RF_CHD and LR_RF_CHD_12u: Creation of the logisitic regression and random forest models. The 12u extension file was used to create the models for the 12u results and the static results. All files/ results will be saved in ouput_folder <br />
RNN_LSTM_CHD: Creation and training of the recurrent neural network model <br />
LTSM_CHD: Evaluation of the RNN-LTSM models, all files/ results will be saved in output_folder <br />
<br />
ROC_CURVES and Feature_importance: files to create the ROC plots and the feature importance plots for LR/ random forest models <br />
<br />
All standard variables, all own functions and attention_function consists of supporting variables and functions for the other files. <br />
<br />
**License** <br />
This project is licensed under the MIT License - see the LICENSE.md file for details

**Acknowledgements** <br />
Thanks to Deepak A. Kaji and Philippe remy for providing open source code for a recurrent neural network and attention function
