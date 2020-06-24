# usage: python scripts/convert_json_to_csv.py -i dataset/carla/055c97917c302ffc6550f5a17ca24be1976398bb/ -o dataset/carla_csv_test

import argparse
import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tqdm

def json_to_csv(input_dir, output_dir, file_size=50, convert_limit=0, visualize=False):
    """
    combine json files to csv files
    :param input_dir: directory of input json files spawned from carla simulator
    :param output_dir: directory of output csv files
    :param file_size: number of json files contained in each csv file
    :param convert_limit: maximum number of files to save. (0 means save all files)
    :return: none
    """
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # get input files
    if os.path.exists(input_dir):
        json_file_list = glob.glob(os.path.join(input_dir,'*.json'))
    else:
        print('Input directory not available: {}'.format(input_dir))

    json_file_list.sort()
    track_counter = 0
    num_json_files = len(json_file_list)
    num_csv_files = len(json_file_list) // file_size
    print('Converting {} json files to {} csv files'.format(num_json_files, num_csv_files))

    for i in tqdm.tqdm(range(num_csv_files)):
        timestamps = []
        pos_x = []
        pos_y = []
        city = ['NA']*file_size
        track_id = ['0']*file_size
        object_type = ['AGENT']*file_size
        for j in range(i*file_size, (i+1)*file_size):
            with open(json_file_list[j], 'r') as f:
                data = json.load(f)
            timestamps.append(data['timestamp'])
            pos_x.append(data['location'][0])
            pos_y.append(data['location'][1])

        # save to csv file
        # headers: TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME
        data_dict = {'TIMESTAMP': timestamps, 'TRACK_ID':track_id, 'OBJECT_TYPE':object_type,
               'X': pos_x, 'Y': pos_y, 'CITY_NAME': city}
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        fname = '{}/{}.csv'.format(output_dir, i)
        df = pd.DataFrame(data_dict)
        df.to_csv(fname, header=True, index=False)

    import IPython; IPython.embed(header='done converting')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input_dir', help='input json files to read')
    parser.add_argument('-o', '--output_dir', help='clustered csv to write')
    parser.add_argument('-n', '--file_size', type=int, default=50, help='number of json files contained in each csv file')
    parser.add_argument('-l', '--limit', type=int, default=0, help='maximum number of csv files to save')
    parser.add_argument('-v', '--visualize', default=False, help='visualize tracks')

    args = parser.parse_args()

    json_to_csv(input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_size=args.file_size,
        convert_limit=args.limit,
        visualize=args.visualize)

if __name__ == '__main__':
    main()
