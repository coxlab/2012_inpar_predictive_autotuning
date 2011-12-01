"""
For a given patience in obtaining each plan,
how many gigaflops can you get on average from a particular problem space?
"""

import cPickle
import logging
import sys
import time

import numpy
import pycuda._driver

import wisdom
from hyperopt.ht_dist2 import one_of, rSON2
import fbconv3_cuda

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from pycuda import driver
def init_cuda(dev_id):
    driver.init()
    logger.info( "GPU Device listing")
    for i in range(driver.Device.count()):
        device = driver.Device(i)
        logger.info( "Device %i: %s %s" % (i,  device.name(),
            device.compute_capability()))
    device = driver.Device(dev_id)
    logger.info("Using: %s" % device.name())
    return device


# XXX : should the average GFLOP/S be measured by dividing by trials or by
#       time? (Should it be more important to tune the more expensive calls? I
#       think yes)

def problem_generator(rng):
    # TODO: sample fbcorr parameters from within LFW models
    space = rSON2(
            "nimgs" , 1, #one_of(1, 2, 4, 8, 16),
            "iheight" , one_of(8, 16, 32, 64, 96, 121, 160, 200, 256),
            "iwidth" , one_of(8, 16, 32, 64, 96, 121, 160, 200, 256),
            "depth" , one_of(1, 4, 8, 16, 32, 64), # XXX: 3 for rgb
            "nfilters" , one_of(1, 4, 8, 16, 32, 64), # must be 1 or 4k
            "fsize" , one_of(3, 5, 7, 9, 11),
            )
    while True:
        s = space.sample(rng=rng)
        prob_spec = wisdom.ProblemSpec(
                n_imgs=s['nimgs'],
                height=s['iheight'],
                width=s['iheight'], # XXX: why is assert height==width in FilterOp??
                depth=s['depth'],
                n_filters=s['nfilters'],
                filter_height=s['fsize'],
                filter_width=s['fsize'],
                img_strides=None,
                filter_strides=None,
                border_mode='valid')
        if prob_spec.gflops() > 100:
            # too big...
            continue
        if prob_spec.out_height <= 0:
            continue
        yield prob_spec

def main_step():
    _python, _cmd, dev_id_str, wisdomfile, patience_str = sys.argv

    device = init_cuda(int(dev_id_str))

    try:
        wdb, results, rng = cPickle.load(open(wisdomfile))
    except (IOError, EOFError):
        wdb, results, rng = wisdom.Wisdom(), [], numpy.random.RandomState(2)

    for iii in xrange(1000):
        pgen = problem_generator(rng)
        prob_spec = pgen.next()

        print prob_spec
        if len(wdb._observations) > 3 + getattr(wdb, '_dtree_n_obs', 0):
            wdb.build_dtree(force=True)
        print 'n_observations', len(wdb._observations)

        #
        # The strategy for updating the training set seems to be important.
        # Currently, what seems to work best is that the speed of the
        # suggestion from plan is ALWAYS fed back as a training example, but
        # the feedback of other suggestion mechanisms (ref, random, etc.) is
        # only fed into the training set if it's an improvement over the
        # current best suggestion. Therefore only errors are corrected, and
        # the training set stays "focused?"
        #  Not sure if this is good or not.

        #
        # XXX: how does pycuda's cache interact with this idea of a walltime
        # of patience?
        #

        finding = {}
        for k in 'ref', 'slow', 'wise':
            print "EXP: PLANNING ", k
            if k == 'ref':
                op_spec=wisdom.reference_op_spec()
            elif k == 'quick':
                op_spec=prob_spec.plan(patience=-1, wisdom=None,
                    device=device,
                    rng=rng)
            elif k == 'slow':
                op_spec=prob_spec.plan(patience=float(patience_str), wisdom=None,
                    device=device,
                    rng=rng)
            elif k == 'wise':
                op_spec=prob_spec.plan(patience=float(patience_str),
                    wisdom=wdb,
                    device=device,
                    rng=rng)
            print "EXP: MEASURING ", k, "..."
            speed = prob_spec.measure_speed(op_spec,
                    n_warmups=2, n_runs=8, wisdom=wdb, device=device)
            print "EXP: MEASURED ", speed
            finding[k] = speed

        print 'FINDING', finding
        results.append(finding)

        ofile = open(wisdomfile, 'w')
        cPickle.dump((wdb, results, rng), ofile)
        ofile.close()


def main_insert_random_stuff():
    _python, _cmd, wisdomfile, N = sys.argv

    try:
        wdb = cPickle.load(open(wisdomfile))
    except (IOError, EOFError):
        wdb = wisdom.Wisdom()

    patience = 20  # seconds
    for i, prob_spec in zip(range(int(N)), problem_generator()):
        try:
            random_op_spec = wisdom.random_op_spec(numpy.random)
            random_speed = prob_spec.measure_speed(random_op_spec,
                    n_warmups=2, n_runs=6)
            break
        except fbconv3_cuda.InvalidConfig:
            random_speed = 0
            continue
        except pycuda._driver.LogicError:
            #XXX: cuModuleGetTexRef not found
            random_speed = 0
            continue
        except pycuda._driver.CompileError:
            #XXX: cuModuleGetTexRef not found
            random_speed = 0
            continue
        print 'RANDOM:', random_speed
        wdb.record(prob_spec, random_op_spec, random_speed)

    cPickle.dump(wdb, open(wisdomfile, 'w'))


def main_dtree():
    _python, _cmd, wisdomfile = sys.argv
    wdb = cPickle.load(open(wisdomfile))
    wdb.build_dtree()
    cPickle.dump(wdb, open(wisdomfile, 'w'))


def main_fig1():
    _python, _cmd, wisdomfile = sys.argv
    wdb, results, rng = cPickle.load(open(wisdomfile))
    import matplotlib.pyplot as plt
    for key, col in ('slow', 'r'),  ('wise', 'g'), ('quick', 'b'):
        y = [r[key] / r['ref'] if (r['ref'] > 0 and r[key]>0) else 1
                for r in results]
        print y
        plt.scatter(numpy.arange(len(y)), y, label=key, c=col)
        yy = [r[key] / r['ref']
                for r in results if r['ref'] > 0 and r[key] > 0]
        gmean = numpy.exp( numpy.log(yy).mean())
        plt.axhline(gmean, c=col)

    plt.xlabel('amount of training data')
    plt.ylabel('speed up over reference')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    cmd = sys.argv[1]
    main = globals()['main_' + cmd]

    sys.exit(main())
