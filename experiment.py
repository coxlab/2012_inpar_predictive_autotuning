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
            "isize" , one_of(8, 13, 27, 28, 32, 81, 100, 121, 160, 200, 256),
            "depth" , one_of(1, 4, 8, 16, 32, 64), # XXX: 3 for rgb
            "nfilters" , one_of(1, 4, 8, 16, 32, 64), # must be 1 or 4k
            "fsize" , one_of(3, 5, 7, 9, 11),
            )
    while True:
        s = space.sample(rng=rng)
        prob_spec = wisdom.ProblemSpec(
                n_imgs=s['nimgs'],
                height=s['isize'],
                width=s['isize'],
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
    _python, _cmd, dev_id, wisdomfile, N = sys.argv

    device = init_cuda(int(dev_id))

    try:
        wdb, results, rng = cPickle.load(open(wisdomfile))
    except (IOError, EOFError):
        wdb, results, rng = wisdom.Wisdom(), [], numpy.random.RandomState(2)

    try:
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
        #

        smart_op_spec = prob_spec.plan(patience=-1,
                wisdom=wdb,
                verbose=1,
                device=device
                )
        ref_op_spec = wisdom.reference_op_spec()

        smart_speed = prob_spec.measure_speed(smart_op_spec,
                n_warmups=2, n_runs=5,
                wisdom=wdb)

        if smart_op_spec != ref_op_spec:
            ref_speed = prob_spec.measure_speed(ref_op_spec,
                    n_warmups=2, n_runs=5,
                    wisdom=None)
            if ref_speed > smart_speed:
                wdb.record(prob_spec, ref_op_spec, ref_speed)
            finding = dict(
                    smart=smart_speed,
                    ref=ref_speed)
            results.append(finding)
            best_op_spec = smart_op_spec if smart_speed > ref_speed else ref_op_spec
        else:
            results.append(dict(smart=smart_speed, ref=smart_speed))
            best_op_spec = smart_op_spec

        print 'FINDING', results[-1]
        # -- some exploration
        N = float(N)
        while N > 0:
            random_op_spec = wisdom.random_op_cross(best_op_spec,
                    wisdom.random_op_spec(rng),
                    rng, .75)

            if random_op_spec == best_op_spec:
                N -= .2
                continue
            random_speed = prob_spec.measure_speed(random_op_spec,
                n_warmups=2, n_runs=5,
                wisdom=None,
                abort_thresh=smart_speed * .75, # should be best speed
                save_on_abort=False)
            if random_speed > smart_speed:
                wdb.record(prob_spec, random_op_spec, random_speed)
            print random_speed
            N -= 1
    finally:
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
    y = [r['smart'] / r['ref'] for r in results if r['ref'] > 0]
    plt.scatter(numpy.arange(len(y)), y)
    plt.xlabel('amount of training data')
    plt.ylabel('speed of dtree / speed of reference')
    plt.axhline(1.0)
    plt.show()


if __name__ == '__main__':
    cmd = sys.argv[1]
    main = globals()['main_' + cmd]

    sys.exit(main())
