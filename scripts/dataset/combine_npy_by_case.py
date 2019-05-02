from __future__ import print_function
import os
import re
import sys
import glob
import argparse
import numpy as np

"""
Scan a directory and join all the npy arrays from the same case,
- cases defined by the `SP*` code prepending the filename.

"""
from CASE_LABEL_DICT import CASE_LABEL_DICT

CASE_PATT = r'(?P<cname>SP.\d+.\d+).+'
def get_case_names(file_list):
    file_bases = [os.path.basename(x) for x in file_list]
    print(file_bases)
    case_names = []
    # case_names = [re.findall(CASE_PATT, x)[0] for x in file_bases]
    for x in file_bases:
      cn = re.findall(CASE_PATT, x)[0]
      cn = cn.replace(' ', '_')
      print(x, cn)
      case_names.append(cn)

    return np.asarray(case_names)


def main(args, error_file):
    file_list = np.asarray(sorted(glob.glob(os.path.join(args.npy_dir, '*.npy'))))
    case_list = get_case_names(file_list)
    unique_cases = np.unique(case_list)

    print('Files: ', len(file_list))
    print('Cases: ', len(unique_cases))

    case_files = {}
    for case_name in unique_cases:
        try:
            case_label = CASE_LABEL_DICT[case_name]
        except:
            error_str = '++ LABEL ERRROR CASE {}\n'.format(case_name)
            error_file.write(error_str)
            continue

        case_index = case_list == case_name
        case_files[case_name] = file_list[case_index]

    n_tiles = []
    for case_name in case_files.keys():
        case_label = CASE_LABEL_DICT[case_name]
        out_file = os.path.join(args.out_dir, case_name)
        fx = case_files[case_name]
        print('\nCase: {} ({})'.format(case_name, case_label))
        print('Input:')

        np_out = []
        for fx_path in fx:
            fx_data = np.load(fx_path, mmap_mode='r')
            n_tiles.append(fx_data.shape[0])
            np_out.append(fx_data)
            print('\t{}: {}'.format(fx_path, fx_data.shape))

        print('Ouptut:')
        np_out = np.concatenate(np_out, axis=0)
        print('\t{}: {}'.format(out_file, np_out.shape))

        np.save(out_file, np_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_dir')
    parser.add_argument('out_dir')

    args = parser.parse_args()
    with open('tile_data_wrangling_errors.txt','w+') as error_file:
      main(args, error_file)
