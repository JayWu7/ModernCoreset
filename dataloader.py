import os
import shutil
import urllib.request as request
import zipfile
import pandas as pd
import numpy as np

# Data list, you can add your data that just follow the same typing forms: name: url
data_list = {
    'Activity recognition exp': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip'
}


def download(filename, data_root='./data'):
    assert filename in data_list, 'Please add the filename and url in data_list above.'
    url = data_list[filename]
    filepath = os.path.join(data_root, filename)
    request.urlretrieve(url, filepath)
    assert os.path.exists(filepath), 'Download fail!, please try again'
    zip_data = zipfile.ZipFile(filepath)
    zip_data.extractall()
    zip_data.close()
    os.remove(filepath)
    shutil.move('./{}'.format(filename), data_root)

    return True


def existed_check(filename):
    if 'data' not in os.listdir():
        os.makedirs('./data')

    if filename not in os.listdir('./data'):
        print("Data {} doesn't exist in local, download starting!".format(filename))
        if download(filename):
            print('Data download successfully!')
        else:
            pass  # May add something in the future

    print('{} check finished!'.format(filename))


def data_filter(data):
    '''
    Function to filter the original data and discard the none float or integral type columns
    :param data: input data
    :return: numerical data after filtering
    '''
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    return numerical_data

def loader(filename='Activity recognition exp', specific_file=None):
    existed_check(filename)
    if specific_file:  # When filename is a directory not a data file.
        filename = os.path.join(filename, specific_file)
    filepath = os.path.join('./data', filename)
    data = pd.read_csv(filepath)
    nm_data = data_filter(data)
    return nm_data.to_numpy()


def sample(data, size=1000):
    sample_idx = np.random.choice(len(data), size=size)
    return data[sample_idx]


if __name__ == '__main__':
    data = loader(filename='Activity recognition exp', specific_file='Watch_gyroscope.csv')
    print(data.shape)
    print(sample(data))


