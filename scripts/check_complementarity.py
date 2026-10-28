"""Evaluate fixed equal-weight ConvNeXt blends; write aggregate statistics only."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from compare_predictions import read_predictions
from glaucoma.fusion import align_by_name, paired_bootstrap, ranking_metrics, sigmoid, logit


def compare(first, second, reps=2000):
    a,b=align_by_name([first,second])
    y=np.asarray(a['targets'],dtype=int)
    if not np.array_equal(y,b['targets']):
        raise ValueError('Prediction labels differ')
    scores=[np.asarray(p['logits'] if 'logits' in p else logit(p['probabilities'])) for p in (a,b)]
    if any(s.shape!=y.shape or not np.isfinite(s).all() for s in scores):
        raise ValueError('Invalid prediction scores')
    negative=scores[0][y==0],scores[1][y==0]
    # Each positive's count of inverted positive-negative pairs localizes AUC loss.
    inversion=[((s[y==1,None] < n[None,:]).sum(axis=1)
                + .5*(s[y==1,None] == n[None,:]).sum(axis=1)) for s,n in zip(scores,negative)]
    result={'count':len(y),'positives':int(y.sum()),
            'spearman':float(spearmanr(*scores).statistic),
            'first':ranking_metrics(y,scores[0]),'second':ranking_metrics(y,scores[1]),
            'positive_inversions':{'first_total':float(inversion[0].sum()),
                                   'second_total':float(inversion[1].sum()),
                                   'second_better_count':int((inversion[1]<inversion[0]).sum()),
                                   'first_better_count':int((inversion[0]<inversion[1]).sum())},
            'blends':{},
            'interpretation':'Fixed equal weights; no validation weight search. Same validation used for checkpoint selection; bootstrap does not measure training-seed variability.'}
    for name, blended in [('equal_raw_logits',(scores[0]+scores[1])/2),
                          ('equal_probabilities',(sigmoid(scores[0])+sigmoid(scores[1]))/2)]:
        result['blends'][name]={'metrics':ranking_metrics(y,blended),
            'paired_vs_first':paired_bootstrap(y,blended,scores[0],n=reps,metrics=('auroc','ap'))}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first');parser.add_argument('second')
    parser.add_argument('--output',required=True);parser.add_argument('--reps',type=int,default=2000)
    args=parser.parse_args()
    result=compare(read_predictions(args.first),read_predictions(args.second),args.reps)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
