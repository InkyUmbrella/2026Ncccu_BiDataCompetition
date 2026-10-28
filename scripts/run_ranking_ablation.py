"""Compare natural BCE, sampled BCE and sampled BCE+ranking on one machine."""
import argparse

from run_resolution_ablation import main


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--recipes',nargs='+',choices=('natural','sampled','rank'),
                        default=['sampled','rank'])
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--workers',type=int,default=0,help='Windows-safe default; choose before starting')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--image-cache',help='Existing compatible 512 cache; shared between arms')
    parser.add_argument('--cache',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if len(set(args.recipes))!=len(args.recipes):parser.error('Do not repeat recipes')
    if args.cache and args.image_cache:parser.error('Choose --cache or --image-cache')
    args.recipe_family='ranking';args.recipe_names=tuple(args.recipes)
    main(args)
