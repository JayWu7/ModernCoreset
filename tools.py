import matplotlib.pyplot as plt


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


def plot_1(arrays, labels):
    le = len(arrays[0])

    plt.figure(figsize=(10, 6))
    plt.title('Coreset Relative Error Changes with the Coreset Size')
    plt.xlabel('Coreset Size')
    plt.ylabel('Relative Error')
    plt.plot(arrays[0], label='Watch_gyroscope', color='orange')
    plt.plot(arrays[1], label='USCensus1990', color='black')
    plt.plot(arrays[2], label='HK-osm', color='royalblue')
    plt.plot([0 for _ in range(le)], '--', label='Zero Line', color='lightgrey')
    plt.legend()

    plt.xticks([i for i in range(le)], labels, rotation=30)

    # plt.show()
    plt.savefig('plot_1.png', dpi=800)


def plot_2(arrays, labels):
    le = len(arrays)

    plt.figure(figsize=(10, 6))
    plt.title('Coreset Construction Speed Changes with the Input Data Size')
    plt.xlabel('Input Data Size')
    plt.ylabel('Coreset Construction Time (s)')
    plt.plot(arrays, label='Coreset Speed', color='black')

    plt.legend()

    plt.xticks([i for i in range(le)], labels, rotation=30)

    # plt.show()
    plt.savefig('plot_2.png', dpi=800)


if __name__ == '__main__':
    # write_list([1, 2, 3], './a.txt')
    # labels = [100, 200, 500, 800, 1000, 2000, 5000, 8000, 10000, 20000, 50000, 80000, 100000]
    # res_1 = [0.024085972, 0.001072616, -0.008699582, -0.002574244, -0.009112084, -0.011478116, -0.025171562,
    #          -0.03415376, -0.011187217, -0.0252848225, -0.027707808, -0.027514924, -0.013150216]
    # res_2 = [0.046315499, 0.028951899, 0.00244828599, 0.002624942, -0.00292165, -0.01968584378, -0.030666252,
    #          -0.029650818, -0.020176628, 0.00228144, -0.00930038, -0.01138555, -0.026102374]
    # res_3 = [0.0354974, 0.0157664, 0.0155286, 0.0155286, 0.00642237, 0.0157664, 0.0155286, 0.0155286, 0.00642237,
    #          0.0157664, 0.0155286, 0.0155286, 0.00642237]
    #
    # plot_1([res_1, res_2, res_3], labels)
    res = [0.0001, 0.01, 0.01, 0.02, 0.03, 0.165, 3.49, 16.44, 32.72, 163.035, 327.55]
    labels = [10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000,500000000,1000000000]

    plot_2(res, labels)

