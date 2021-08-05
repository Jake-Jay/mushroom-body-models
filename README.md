# Modelling the Fruit Fly Mushroom Body

The Mushroom Body (MB) is established as a site for associative learning in insects. The structure is split into 15 compartments, demarcated by the presence of specific mushroom body output neurons (MBONs), and dopamin- ergic neurons (DANs). Each of these com- partments provides a site for parallel learning. Dopamine has been shown to convey essential information about the valence of odours to each of these compartments. Similarly it has been shown that MBONs use this valency informa- tion in their representation of odours to initiate an appropriate behavioural response. We sought to model the relationship between DANs and MBONs. Several models are investigated rang- ing from a simple one-to-one linear model to a recurrent model where MBONs feed back into the MB compartments. Given that the connec- tome of the Drosophila Melanogaster MB has recently become available, the models’ learned weights are compared with these known synap- tic connections.

# Most Relevent notebooks:

Note that all paths in this document are given as relative paths from the Notebook in which they are called and NOT from the project root directory.

The Notebooks should be run in order since they often depend on the output from the previous notebook.

- `siju-PlotAll-modified.ipynb`
    - input: `../data/experiment_summary.h5`
    - must rerun this but I think that it generates '../data/siju-dataframe-unprocessed.pkl'

- `preprocess-siju`
    - input: '../data/siju-dataframe-unprocessed.pkl'
    - generates: '../data/siju-cleaned-expanded-pivot.pkl'
    
- `preprocess-hige-expanded.ipynb`
    - input: '../data/hige_mbon_response.mat'
    - generates: '../data/hige-cleaned-expanded-pivot.pkl'

- `build-datadistributions.ipynb`
    - description: create a set of distributions to model the activity in each compartment for each stimuli for both DANs and MBONs (this is to increase the size of the given dataset).
    - inputs:
        - '../data/siju-cleaned-expanded-pivot.pkl'
        - '../data/hige-cleaned-expanded-pivot.pkl'
    - generates: 
        - '../models/siju_lognorm_fits.json'
        - '../models/hige_lognorm_fits.json'

- `build-timeseriesdata.ipynb`
    - description: create train, validation, and test datasets to fit the linear and recurrent models. This uses the distributions generated in `build-datadistributions.ipynb`
    - inputs:
        - '../models/siju_lognorm_fits.json'
        - '../models/hige_lognorm_fits.json'
    - generates:
        - Scalar data (for linear models):
            - '../data/X-train-from-distribution.npy'
            - '../data/X-val-from-distribution.npy'
            - '../data/X-test-from-distribution.npy'
            - '../data/Y-train-from-distribution.npy'
            - '../data/Y-val-from-distribution.npy'
            - '../data/Y-test-from-distribution.npy'
        - Time-series data (for recurrent model)
            - '../data/X-train-exp-time-series-from-distribution.npy'
            - '../data/X-val-exp-time-series-from-distribution.npy'
            - '../data/X-test-exp-time-series-from-distribution.npy'            
            - '../data/Y-train-exp-time-series-from-distribution.npy'
            - '../data/Y-val-exp-time-series-from-distribution.npy'
            - '../data/Y-test-exp-time-series-from-distribution.npy'
        

## Linear Models
- `linear-models-sampled-data.ipynb`
    - description: fits the data to a simpe linear model and a linear model with cross talk between MB lobes
    - inputs:
        - '../data/X-train-from-distribution.npy'
        - '../data/X-test-from-distribution.npy'            
        - '../data/Y-train-from-distribution.npy'
        - '../data/Y-test-from-distribution.npy'
    - generates:

## Recurrent Models
- `connectomics.ipynb`
    - description: connectomics data from Li et al 2020
    - inputs: none
    - generates: `../data/mbon-connectomics.npy`


- `train_masked_rnn.ipynb`
    - description: train an RNN but constrain the linear layers with masking matrices based on connectomics data
    - inputs: 
        - '../data/X-train-exp-time-series-from-distribution.npy'
        - '../data/X-val-exp-time-series-from-distribution.npy'
        - '../data/X-test-exp-time-series-from-distribution.npy'            
        - '../data/Y-train-exp-time-series-from-distribution.npy'
        - '../data/Y-val-exp-time-series-from-distribution.npy'
        - '../data/Y-test-exp-time-series-from-distribution.npy'
        - `../mbon-connectomics.npy`
    - generates:

# Contributions

This research was conducted as part of an internship at the Max Plank Institute for Brain Research in Julijana Gjorgieva's group. It was supervised by Prof. Gjorgieva and Shuai Shao.
