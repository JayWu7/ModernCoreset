

def write_list(list_data, output_path):
    '''
    This is a functional method to write the python list into the text file.
    :param list_data: data of list
    :param output_path: output path to the text file
    '''
    assert hasattr(list, '__iter__'), "Please ensure the input data is iterable!"
    with open(output_path, 'w') as f:
        for row in list_data:
            f.write(str(row) + '\n')


if __name__ == '__main__':
    write_list([1, 2, 3], './a.txt')
