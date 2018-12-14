"""
Reduce the number of tiles per case to some constant,
Also obfuscate the original file names 
"""
from __future__ import print_function

import numpy as np
import hashlib
import pickle
import glob
import os

from CASE_LABEL_DICT import CASE_LABEL_DICT

src_list = glob.glob('./tiles/*.npy')
dst_dir = 'tiles_reduced'
target = 1000
case_dict = {}

print('Got {} source files'.format(len(src_list)))
for i,src in enumerate(src_list):
    case_name = os.path.basename(src).replace('.npy', '')
    case_label = CASE_LABEL_DICT[case_name]

    md5hash = hashlib.md5(case_name.encode()).hexdigest()

    case_dict[md5hash] = case_label
    dst_name = os.path.join(dst_dir, '{}.npy'.format(md5hash))
    
    x = np.load(src)
    if x.shape[0] < target:
        np.save(dst_name, x)

    else:
        indices = np.random.choice(range(x.shape[0]), target, replace=False)
        x = x[indices]
        np.save(dst_name, x)

    print('{:03d} {}\t{}\t{}\t{}'.format(i, md5hash, case_label, dst_name, x.shape))

with open('case_dict_obfuscated.pkl', 'wb') as f:
    pickle.dump(case_dict, f)
