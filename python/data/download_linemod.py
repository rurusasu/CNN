import os
from multiprocessing import Pool
import urllib.request

"""
LINEMOD Dataset 製作者の Stefan Hinterstoißer さんのサイトから
LINEMOD Dataset の元データをダウンロードする。
"""


object_names = ['ape', 'benchviseblue', 'bowl', 'cam', 'can', 'cat',
                'cup', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
                'iron', 'lamp', 'phone']

# Stefan Hinterstoißer さんのサイトのURL
base_url = 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/{}.zip'
target_dir = 'temp'


def download_and_unzip(object_name):

    url = base_url.format(object_name)
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(
        target_dir, 'linemod_{}.zip'.format(object_name))
    urllib.request.urlretrieve(url, target_file)
    os.makedirs('linemod', exist_ok=True)
    os.system('unzip {} -d linemod/original_dataset'.format(target_file))


if __name__ == "__main__":
    for object_name in object_names:
        download_and_unzip(object_name)
