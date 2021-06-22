# metalearning_compression

Contains training scripts for image compression + metalearning experiments. 

**FiLM_baseline_model.py:** Defines baseline BMSHJ2018Model 

**FiLM_baseline_train.py:** Trains baseline BMSHJ2018Model 

**FiLM_baseline_model.py:** Defines BMSHJ2018Model with context (conditional model)

**FiLM_cond_train.py:** Trains BMSHJ2018Model with context (conditional model)

**FiLM_overfit.py:** Finetunes/overfits trained baseline BMSHJ2018Model to specific datasets

**FiLM_two_stage.py:** Trains context by model by first loading trained baseline model (first stage), and only training context model + conditional layers (second stage)

**data_helpers.py:** Helper functions for tensorflow dataset loading/creation
