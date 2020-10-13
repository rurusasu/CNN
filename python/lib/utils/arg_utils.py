import argparse


def parse_args():
    parser = argparse.ArgumentPaser()

    # debug
    parser.add_argument('--debug', action='store_true',
                        dest='debug', default=False)
    parser.add_argument('--rec', action='store_true',
                        dest='rec', default=False)
    parser.add_argument('--rec_name', default='resnet', type=str)
    parser.add_argument('--net', action='store', dest='net',
                        type=str, choices=['resnet'])
    parser.add_argument('--type', action='store', dest='type', type=str)

    # detaset
    parser.add_argument('--dataset', action='store', dest='dataset',
                        type=str, choices=['linemod', 'occlustion_linemod'])

    args = parser.parse_args()

    return args
