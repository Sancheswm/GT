import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--reg', default=1e-2, type=float, help='weight decay regularizer')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
    parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
    parser.add_argument('--latdim', default=16, type=int, help='embedding size')
    parser.add_argument('--memosize', default=2, type=int, help='memory size')
    parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
    parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--shoot', default=10, type=int, help='K of top k')
    parser.add_argument('--data', default='ijcai', type=str, help='name of dataset')
    parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
    parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')
    parser.add_argument('--mult', default=1e2, type=float, help='multiplier for the result')
    parser.add_argument('--keepRate', default=0.7, type=float, help='rate for dropout')
    parser.add_argument('--iiweight', default=0.3, type=float, help='weight for ii')
    parser.add_argument('--slot', default=5, type=float, help='length of time slots')
    parser.add_argument('--graphSampleN', default=2000, type=int, help='use 25000 for training and 200000 for testing, empirically')
    parser.add_argument('--divSize', default=512, type=int, help='div size for smallTestEpoch')
    parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
    parser.add_argument('--subUsrSize', default=10, type=int, help='number of item for each sub-user')
    parser.add_argument('--subUsrDcy', default=0.9, type=float, help='decay factor for sub-users over time')
    return parser.parse_args()

args = parse_args()

# Dataset-specific settings (moved outside the conditional)
dataset_settings = {
    'jd2021': {'user': 57721, 'item': 4172},
    'tianchi': {'user': 423423, 'item': 874328},
    'tmall': {'user': 805506, 'item': 584050},  # or 147894 and 99037
    'ML10M': {'user': 67788, 'item': 8704},
    'yelp': {'user': 19800, 'item': 22734},
    'ijcai': {'user': None, 'item': None} # Placeholder, will be set later
}

if args.data in dataset_settings:
    settings = dataset_settings[args.data]
    if settings['user'] is not None:  # Only set if values are provided
        args.user = settings['user']
        args.item = settings['item']
elif args.data == 'ijcai': # Handle ijcai separately
    pass # User and item will be set by the data handler
else:
    raise ValueError(f"Unknown dataset: {args.data}")


# Swap user and item (optional, comment out if not needed)
# tem = args.user
# args.user = args.item
# args.item = tem

args.decay_step = args.trnNum // args.batch
