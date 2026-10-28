import unittest
import numpy as np
from test_fusion import load_script


class ComplementarityTests(unittest.TestCase):
    def test_alignment_complementary_errors_and_label_rejection(self):
        script=load_script('check_complementarity')
        a={'names':['a','b','c','d'],'targets':[0,1,0,1],'logits':[0.,-1.,-2.,3.]}
        b={'names':['d','c','b','a'],'targets':[1,0,1,0],'logits':[1.,2.,3.,-2.]}
        result=script.compare(a,b,reps=20)
        self.assertEqual(result['blends']['equal_raw_logits']['metrics']['auroc'],1.)
        self.assertEqual(result['positive_inversions']['second_better_count'],1)
        b['targets'][0]=0
        with self.assertRaises(ValueError):
            script.compare(a,b,reps=20)
