# Most Relevent notebooks:

Note that all paths in this document are given as relative paths from the Notebook in which they are called and NOT from the project root directory.

The Notebooks should be run in order since they oftenrffgffgfgfgtftgtgrtfgtgtgrfg

- `siju-PlotAll-modified.ipynb`
    - input: `../data/experiment_summary.h5`
    - must rerun this but I think that it generates '../data/siju-dataframe-unprocessed.pkl'

- `preprocess-siju`
    - input: '../data/siju-dataframe-unprocessed.pkl'
    - generates: '../data/siju-cleaned-expanded-pivot.pkl'
    
- `preprocess-hije-expanded.ipynb`
    - input: '../data/hije_mbon_response.mat'
    - generates: '../data/hije-cleaned-expanded-pivot.pkl'

- `build-datadistributions.ipynb`
    - description: create a set of distributions to model the activity in each compartment for each stimuli for both DANs and MBONs (this is to increase the size of the given dataset).
    - inputs:
        - '../data/siju-cleaned-expanded-pivot.pkl'
        - '../data/hije-cleaned-expanded-pivot.pkl'
    - generates: 
        - '../models/siju_lognorm_fits.json'
        - '../models/hije_lognorm_fits.json'

- `build-timeseriesdata.ipynb`
    - description: create train, validation, and test datasets to fit the linear and recurrent models. This uses the distributions generated in `build-datadistributions.ipynb`
    - inputs:
        - '../models/siju_lognorm_fits.json'
        - '../models/hije_lognorm_fits.json'
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
        
        
- `connectomics.ipynb`
    - description: connectomics data from Li et al 2020
    - inputs: none
    - generates: `../data/mbon-connectomics.npy`


- 'train_masked_rnn.ipynb'
    - description: train an RNN but constrain the linear layers with masking matrices based on connectomics data
    - inputs: 
        - '../data/X-train-exp-time-series-from-distribution.npy'
        - '../data/X-val-exp-time-series-from-distribution.npy'
        - '../data/X-test-exp-time-series-from-distribution.npy'            
        - '../data/Y-train-exp-time-series-from-distribution.npy'
        - '../data/Y-val-exp-time-series-from-distribution.npy'
        - '../data/Y-test-exp-time-series-from-distribution.npy'
        - `../mbon-connectomics.npy`