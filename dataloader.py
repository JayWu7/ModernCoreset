import os
import shutil
import urllib.request as request
import zipfile
import pandas as pd

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


def loader(filename='Activity recognition exp', specific_file=None):
    existed_check(filename)
    if specific_file:  # When filename is a directory not a data file.
        filename = os.path.join(filename, specific_file)
    filepath = os.path.join('./data', filename)
    return pd.read_csv(filepath)


if __name__ == '__main__':
    loader(filename='Activity recognition exp', specific_file='Watch_gyroscope.csv')


