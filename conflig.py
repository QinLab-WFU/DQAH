import argparse
from crossFormer_config import get_config
def parse_option():


    # easy config modification
    parser.add_argument('--epochs',default=300)
    parser.add_argument('--batch-size', type=int,default=64, help="batch size for single GPU")
    parser.add_argument('--data-set', type=str, default='imagenet', help='dataset to use')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume',default='', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='native', choices=['native', 'O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--num_workers', type=int, default=8, help="")
    parser.add_argument('--mlp_ratio', type=int, default=4, help="")
    parser.add_argument('--warmup_epochs', type=int, default=20, help="#epoches for warm up")
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')
    
    
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config