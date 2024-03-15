# Environmental setup
```conda env create -f requirements.yaml```

# STpGCN for brain decoding
From the project's root folder, run

```python main.py```

to train the **STpGCN** or **STpGCN-alpha** or **STpGCN-beta** or **STpGCN-gamma** or **STGCN** or **GAT** or **GCN** or **GIN** or **MLP-Mixer** on the HCP S1200 datasets.

It would help if you manually changed the *model_name* to ensure you train the desired model.

# NeurocircuitX for explainability
Run `python NeurocircuitX_sep.py` to calculate the **importance score** under *keeping* and *masking* strategy for each task.

After finishing the calculation of the importance score, you can run ```python NeurocircuitX_mix.py``` 
to calculate the **final importance score** for each task.

# Main functions
- ```main.py:```: The script serves as the entry point for the fMRI-based brain decoding pipeline. It orchestrates the entire process, from data loading and preprocessing to model training and evaluation.

- ```stpgcn_variants.py```: The following spatial-temporal model variants are implemented:

    1. STGCN: The basic STGCN model with a configurable structure using the control_str parameter.
    2. STpGCN: STGCN with a pyramid structure for multi-scale temporal modelling.
    3. STGCN_hidden_feature: STGCN variant that outputs hidden features instead of final predictions.
    4. STpGCN_ab_bottom_up: STpGCN variant without a bottom-up pathway.
    5. STpGCN_ab_top: STpGCN variant without top pathway.
    6. ...

- ```layers.py:```: This file contains the implementation of neural network layers used in `stpgcn_variants.py`.
