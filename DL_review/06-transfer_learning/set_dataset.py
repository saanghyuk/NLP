import requests 
from os import listdir, mkdir, rename
from os.path import isfile, isdir, join
from zipfile import ZipFile


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    print('Downloading zip file completed.')


def unzip(zip_path, dataset_path):
    zf = ZipFile(zip_path)
    zf.extractall(path=dataset_path)
    zf.close()
    print('Unzipping completed.')


def restructure_dir(data_path, is_train=True):
    files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    if is_train:
        for file in files:
            if not isdir(join(data_path, file.split('.')[0])):
                mkdir(join(data_path, file.split('.')[0]))
            rename(
                join(data_path, file), join(data_path, file.split('.')[0], file)
            )
    else:
        for file in files:
            if not isdir(join(data_path, 'dummy')):
                mkdir(join(data_path, 'dummy'))
            rename(
                join(data_path, file), join(data_path, 'dummy', file)
            )
    print('Resturcturing completed.')


if __name__ == '__main__':

    # make dataset directory
    dataset_path = './dataset'
    if not isdir(dataset_path):
        print('Making dataset directory on {}'.format(dataset_path))
        mkdir(dataset_path)

    # set hymenoptera dataset
    hymenoptera_url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
    hymenoptera_path = './hymenoptera.zip'

    download_url(hymenoptera_url, hymenoptera_path)
    unzip(hymenoptera_path, dataset_path)
    rename(join(dataset_path, 'hymenoptera_data'), join(dataset_path, 'hymenoptera'))
    rename(join(dataset_path, 'hymenoptera', 'val'), join(dataset_path, 'hymenoptera', 'test'))

    # # set catdog train dataset
    # catdog_path = join(dataset_path, 'catdog')
    # catdog_train_path = join(catdog_path, 'train')

    # catdog_train_url = 'https://storage.googleapis.com/kagglesdsdata/competitions/3362/31148/train.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1592316104&Signature=ZTemdU4DNJlQttkMiiZdCxHJsRVynHBZWdcLMXpj7Y982mfEPb%2F3Rs8qQy%2FcqRyg6qlSpjThraOHX6%2FEmQX7FxiCEdKsknTJOI6xWWb52%2BaQzDq1EKVRI%2F05A7%2B87gqVp7dmYV%2B6kjY8V801Hgzjg5euOoYA4CiXLoN6e5aBWi%2B7nVqiDvmTzn8WmLtd3%2Fsv86M%2BuuAuYumL0qNEZReQdoixBQmAgjdHMhJoKwNmx07%2FsY7r34sMUcQXE1qg4eyiTT74Ohm17W7Dvy%2FvEI8%2F5Ko5aZ1vaGgwUO8aBLZirpCJ%2Fqi04cJ4AhFaG9qHnYdneIf6JtcT0OstsU5tYSGrZg%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain.zip'
    # catdog_train_zip = './catdog_train.zip'

    # download_url(catdog_train_url, catdog_train_zip)
    # unzip(catdog_train_zip, catdog_path)
    # restructure_dir(catdog_train_path, is_train=True)

    # # set catdog test dataset
    # catdog_test_path = join(catdog_path, 'test')

    # catdog_test_url = 'https://storage.googleapis.com/kagglesdsdata/competitions/3362/31148/test1.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1592324570&Signature=rlsDo6eqMLVryFbJdNbG70KjL3fVELYhyBMq3un3WR0v6%2BfOQ8bGmOcQP%2Flcd%2FsnyhJJKqgiWz6pi5YcC4KdmHIWCZaRjRMhp7hw%2FkZzqFOnDmjCawhaCuxsAmRSxIQfLrFz41EPm7kOgUtU6XqsvHmdxaJ%2Fr%2FwFqo8RAX%2FmOMGFGq0QW7%2BRWXIk9P3r3EqTrV28i2Npvlka8x1EToK1tJX6aqF6CCznW4eE5aRcakiOru45Vi0SUgF5vCpeHBYTUiJkNf8s0wg%2B4NgyqhM77lGZNDkkCfxp%2FBv2yGG%2BL4tL4U1RP1%2F%2FtcOY1M6j0sOLI5Aa4ZWiAGV7zLVNAwJMeA%3D%3D&response-content-disposition=attachment%3B+filename%3Dtest1.zip'
    # catdog_test_zip = './catdog_test.zip'

    # download_url(catdog_test_url, catdog_test_zip)
    # unzip(catdog_test_zip, catdog_path)
    # rename(join(catdog_path, 'test1'), catdog_test_path)
    # restructure_dir(catdog_test_path, is_train=False)
