'''
Data loader modules, support  

follow sklearn format:
- use Bunch (dict with keys exposed as attrs)
- include keys: (data, target, frame, target_names, DESCR, feature_names, filename )
        frame = pandas DataFrame with data, target
'''

import os
from pathlib import Path
import glob
import re
import csv

import pandas as pd
import numpy as np
import scipy

from sklearn.utils import Bunch
from locs import loc177

loadable_path =  Path('data/')

def load_data(read_path=loadable_path):
    '''
    load diffraction dataset
    '''
    rp = Path(read_path)

    # load data
    data = []
    for f in rp.glob('*_data.csv'):
        with open(f) as csv_file:
            data_file = csv.reader(csv_file)
            temp_data = []
            for i, ir in enumerate(data_file):
                temp_data.append(ir)

            data.append(temp_data)
            
    # Convert list to np.array and "flatten" dimension
    data = np.array(data, dtype=np.float64)
    data = data.reshape((data.shape[0]*data.shape[1], -1))

    # load metadata
    meta = pd.DataFrame()
    for f in rp.glob('*_meta.csv'):
        temp_df = pd.read_csv(f)
        meta = meta.append(temp_df, ignore_index=True)

    # generate "feature names" (q-array)
    feature_names = np.linspace( meta['q_min'][0], meta['q_max'][0], 
                                 num=meta['num_points'][0] 
                                )

    return Bunch(data=data,
                 meta=meta,
                 feature_names=feature_names
                )
    
    

def process_XRD_1D(fp, fkey='*_1D.csv', bounds=(1, 8), 
                    num=1000, export_path=loadable_path):
    '''
    Process individual XRD data and collect into csv with each sample in one row
    interpolate in between bounds with fill value of 0

    return arrays
    '''
    # prep metadata
    p = Path(fp)
    root_folder = p.parts[-1]

    meta_df = pd.DataFrame({ 'data_pt':np.array(list(range(177)))+1, 
                             'px':loc177[0], 'py':loc177[1],
                             'dataset': root_folder, 
                             'num_points': 1000,
                             'q_min': bounds[0],
                             'q_max': bounds[1],
                             'filename': 'NA' }
                          )
    meta_df = meta_df.set_index('data_pt')
    
    # arrays 
    qn = np.linspace(bounds[0], bounds[1], num=num)
    data = []

    # for csv in list
    for fn in glob.glob(fp + fkey):
        df = pd.read_csv(fn, names=['q', 'I'])

        q, I = df.T.values

        # interpolate to fix spacing.
        f = scipy.interpolate.interp1d(q, I, bounds_error=False, fill_value=0)
        In = f(qn)

        # grab interesting info ...?
        name = Path(fn).parts[-1]
        data_pt = int(re.search('_(\d{4})_', name).groups()[0])
        meta_df['filename'][data_pt] = name

        # add values to array...
        data.append(In)

        # save to csv

    data = np.array(data)
    
    # save
    np.savetxt(Path(export_path) / f'{root_folder}_data.csv', data, delimiter=',')
    meta_df.to_csv(Path(export_path) / f'{root_folder}_meta.csv')

