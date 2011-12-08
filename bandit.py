import time
from hyperopt.base import Bandit
from hyperopt.ht_dist2 import one_of, rSON2
import fbconv3_benchmark
import fbconv3_utils
import pycuda.driver
import scipy as sp
import numpy as np


class FBCorr3Bandit(Bandit):
    def __init__(self, nimgs, height, width, depth, nfilters, fsize):
        Bandit.__init__(self, template=rSON2(
            'probspec', rSON2(
                "nimgs" , nimgs,
                "height" , height,
                "width" , width,
                "depth" , depth,
                "nfilters" , nfilters,
                "fsize" , fsize,
                ),
            'metaparams', rSON2(
                "block_w" , one_of(4, 8, 16, 32, 64, 128, 256, 512),
                "block_h" , one_of(4, 8, 16, 32, 64, 128, 256, 512),
                "n_filter_rows" , one_of(1, 2),
                "n_output4s" , one_of("all", 1, 2),
                "spill" , one_of(False, True),
                "imul_fast" , one_of(False, True),
                "pad_shared" , one_of(False, True),
                "use_tex1dfetch" , one_of(False, True),
                "maxrregcount" , one_of(None, 8, 16, 20, 24, 28, 32)
                )))

    def vs_theano(self):
        import theano
        probspec = self.template.sample(1)['probspec']
        img_shp = (probspec['nimgs'], probspec['depth'],
                probspec['height'], probspec['width'])
        k_shp = (probspec['nfilters'], probspec['depth'],
                probspec['fsize'], probspec['fsize'])
        imgs = theano.shared(np.random.rand(*img_shp).astype('float32'))
        filts = theano.shared(np.random.rand(*k_shp).astype('float32'))
        outs = theano.shared(np.random.rand(2, 2, 2, 2).astype('float32'))
        x = theano.tensor.nnet.conv.conv2d(imgs, filts,
                img_shp, k_shp, border_mode='valid')
        f = theano.function([], [], updates=[(outs, x)])
        dt = float('inf')
        for i in range(5):
            t = time.time()
            f()
            dt = min(dt, time.time() - t)

        fsize = probspec['fsize']
        depth = probspec['depth']
        gflop = outs.get_value(borrow=True).size * (fsize * fsize * depth * 2) / (1000.**3.)
        rval =  gflop / dt
        print 'GFLOP/S', rval
        return rval


    @classmethod
    def evaluate(cls, config, ctrl):
        probspec = config['probspec']
        metaparams = config['metaparams']
        try:
            timings, gflop = fbconv3_benchmark.benchmark_run(
                    nimgs=probspec['nimgs'],
                    height=probspec['height'],
                    width=probspec['width'],
                    depth=probspec['depth'],
                    n_filters=probspec['nfilters'],
                    fsize=probspec['fsize'],
                    metaparams=metaparams,
                    n_warmups=2,
                    n_runs=10,
                    noverify=False)
        except fbconv3_utils.InvalidConfig:
            print 'InvalidConfig', metaparams
            return dict(loss=0,
                    timings={},
                    status='fail',
                    reason='invalidconfig')
        except pycuda.driver.CompileError:
            print 'CompileError', metaparams
            return dict(loss=0,
                    timings={},
                    status='fail',
                    reason='CompileError')

        #processtime = sp.median(timings['process'])
        processtime = np.min(timings['process'])
        return dict(
                loss=-gflop / processtime,
                status='ok')


def FBCorr3_cherrypick_256():
    return FBCorr3Bandit(
            nimgs=1,
            height=256,
            width=256,
            depth=8,
            nfilters=64,
            fsize=9,)

def FBCorr3_16_256_rgba_f64_7():
    return FBCorr3Bandit(
            nimgs=4,
            height=256,
            width=256,
            depth=4,
            nfilters=64,
            fsize=7,)


def FBCorr3_cherrypick_2048():
    return FBCorr3Bandit(
            nimgs=1,
            height=2048,
            width=2048,
            depth=4,
            nfilters=4,
            fsize=9,)

if __name__ == '__main__':
    bandit = FBCorr3_16_256_rgba_f64_7()
    bandit.vs_theano()
