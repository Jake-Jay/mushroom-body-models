# Description of Datasets

- Raw Datasets:
    - `data/experiments_summary.h5` - this is directly from the Siju repository (*NOTE* I could not read in this file using the library Siju recommended due to Python dependency clashes. Rather use `data/siju-dataframe-unprocessed.pkl`)
    - `data/hije_mbon_response.mat` - hije raw data
    - `data/siju-dataframe-unprocessed.pkl`- raw data in a dataframe (easier to read than h5)
- Cleaned Data
    - Averaged Activity per lobe:
        - `data/siju-cleaned-averaged.pkl` - built in `notebooks/preprocess-siju.ipynb`
        - `data/siju_pivot_table.csv`
        - `data/hije_pivot_table.csv`
    - Expanded Data - cross Product
        - `data/siju-cleaned-expanded-pivot.pkl` - build in `notebooks/preprocess-siju.ipynb`
        - `data/hije-cleaned-expanded-pivot.pkl` - built in `notebooks/preprocess-hije-expanded.ipynb`
        - Data Matrices (numpy objects with no metadata):
            - Both of these are built in `notebooks/build-datamatrix.ipynb`
            - `X-siju-cross-product.npy`
            - `Y-hije-cross-product.npy`