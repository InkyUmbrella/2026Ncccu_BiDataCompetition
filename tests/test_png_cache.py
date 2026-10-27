"""Lossless PNG cache must preserve pixels, augmentation and source checks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from test_baseline import fixture
from glaucoma.data.cache import ImageCache, prepare_cache
from glaucoma.data.dataset import load_rgb, read_samples
from glaucoma.data.transforms import build_transform


class PNGCacheTests(unittest.TestCase):
    def test_png_and_npy_cache_match_original_and_training_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root/'data')
            kwargs=dict(crop_black_border=True,decode='draft')
            prepare_cache(root/'data',root/'npy',32,1,**kwargs)
            prepare_cache(root/'data',root/'png',32,1,storage='png',**kwargs)
            rows=read_samples(root/'data','train')
            npy=ImageCache(root/'npy',rows,32,**kwargs)
            png=ImageCache(root/'png',rows,32,**kwargs)
            for row in rows:
                np.testing.assert_array_equal(np.asarray(npy.load(row.path)),np.asarray(png.load(row.path)))
                original=load_rgb(row.path,'draft',32)
                for training in (False,True):
                    torch.manual_seed(42)
                    source=build_transform(32,training,crop_black_border=True,normalization='imagenet')(original)
                    torch.manual_seed(42)
                    cached=build_transform(32,training,preprocessed=True,normalization='imagenet')(png.load(row.path))
                    torch.testing.assert_close(source,cached,rtol=0,atol=0)
            rows[0].path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'source changed'):
                ImageCache(root/'png',rows,32,**kwargs)

    def test_invalid_png_entry_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root/'data')
            prepare_cache(root/'data',root/'png',32,1,storage='png')
            rows=read_samples(root/'data','train');cache=ImageCache(root/'png',rows,32)
            (root/'png'/rows[0].path.parent.name/(rows[0].path.name+'.png')).write_bytes(b'broken')
            with self.assertRaises(Exception):
                cache.load(rows[0].path)
