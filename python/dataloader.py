import os
import shutil
import urllib.request as request
import zipfile
import numpy as np
import pandas as pd
import requests
from lxml import etree
import csv
from tqdm import tqdm


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


def data_filter(data, filepath):
    '''
    Function to filter the original data and discard the none float or integral type columns
    :param data: input data
    :return: numerical data after filtering
    '''

    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    return numerical_data


def loader(filename='Activity recognition exp', specific_file=None, sep=','):
    existed_check(filename)
    if specific_file:  # When filename is a directory not a data file.
        filename = os.path.join(filename, specific_file)
    filepath = os.path.join('./data', filename)
    data = pd.read_csv(filepath, sep=sep, index_col=0)
    nm_data = data_filter(data, filepath)
    return nm_data.to_numpy()


def sample(data, size=1000):
    sample_idx = np.random.choice(len(data), size=size, replace=False)
    return data[sample_idx]


def download_GDELT(url, directory):
    '''
    This is the functional method to automatically grab the GDELT data through https request
    :param url: the url of download page
    :param directory: the path to store the data
    '''
    r = request.urlopen(url)
    html = etree.HTML(r.read())
    zip_list = html.xpath('//li[position()>2]/a/@href')
    split_idx = url.rfind('/')
    prefix = url[:split_idx]
    for zip in zip_list:
        if 'counts' in zip:
            url = os.path.join(prefix, zip)
            filepath = os.path.join(directory, zip)
            request.urlretrieve(url, filepath)
            assert os.path.exists(filepath), 'Download fail!, please try again'
            zip_data = zipfile.ZipFile(filepath)
            zip_data.extractall(path=directory)
            zip_data.close()
            os.remove(filepath)

    print('Downloading finish!')


def download_osm(place, admin_level, output_file):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
            [out:json];
            area["ISO3166-1"="{}"][admin_level={}];
            (node(area);
            way(area);
            rel(area);
            );
            out center;
            """.format(place, admin_level)

    response = requests.get(overpass_url,
                            params={'data': overpass_query})
    data = response.json()
    coords = []
    for element in tqdm(data['elements']):
        if element['type'] == 'node':
            lon = element['lon']
            lat = element['lat']
            coords.append([lon, lat])
        elif 'center' in element:
            lon = element['center']['lon']
            lat = element['center']['lat']
            coords.append([lon, lat])
    with open(output_file, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        for row in tqdm(coords):
            writer.writerow(row)


def process_osm(osm_path):
    '''
    Processing the osm file, and write it to binary format.
    :param osm_path: Path of osm file
    '''
    for event, element in etree.iterparse(osm_path, tag="node"):
        yield [element.get('lat'), element.get('lon')]
        element.clear()


def write_osm_to_np(osm_path, out_path):
    assert os.path.exists(osm_path), 'OSM file doesn\'t exist!'
    assert os.path.exists(out_path), 'Please give an output file path!'
    osm_data = np.array(list(process_osm(osm_path)))
    assert type(osm_data).__module__ == 'numpy', 'Please input numpy array as data to conduct the evaluation'
    np.save(out_path, osm_data)


def convert_np_to_csv(np_path, csv_path):
    assert os.path.exists(np_path), 'Numpy binary file doesn\' exist!'
    array = np.load(np_path)
    pd.DataFrame(array).to_csv(csv_path, index=False)


if __name__ == '__main__':
    #data = loader(filename='Activity recognition exp', specific_file='Watch_gyroscope.csv')
    # print(data.shape)
    # print(sample(data))
    # url = 'http://data.gdeltproject.org/gkg/index.html'
    # directory = './data/gdelt'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # download_GDELT(url, directory)
    # data = loader(filename='gdelt', specific_file='20200518.gkgcounts.csv', sep='\t')
    # print(data[:10])
    # print(data.shape)
    #download_osm('HK', 3, './data/OSM/hk_osm.csv')
    #data = loader(filename='hayes-roth.csv')
    #print(data)
    for fi in os.listdir('./data'):
        if fi.endswith('.npy'):
            np_path = os.path.join('./data', fi)
            csv_path = os.path.join('./data', fi[:-3] + 'csv')
            convert_np_to_csv(np_path, csv_path)
    #convert_np_to_csv('./data/netherlands-latest.npy', './data/netherlands-latest-latest.csv')
