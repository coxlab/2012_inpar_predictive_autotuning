"""
For a given patience in obtaining each plan,
how many gigaflops can you get on average from a particular problem space?
"""

import cPickle
import logging
import sys
import time

import numpy as np
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
            "nimgs" , 1, #one_of(1, 2, 4, 8, 16, 32, 64, 128),
            "iheight" , one_of(256, 512, 1024, 2048, 4096),
            #"iwidth" , one_of(256, 512, 1024, 2048, 4096),
            "depth" , one_of(1, 4, 8, 16, 32, 64, 128, 256), # XXX: 3 for rgb
            "nfilters" , one_of(1, 4, 8, 16, 32, 64, 128, 256), # must be 1 or 4k
            "fsize" , one_of(3, 5, 7, 9, 11),
            )
    while True:
        s = space.sample(rng=rng)
        prob_spec = wisdom.ProblemSpec(
                n_imgs=s['nimgs'],
                height=s['iheight'],
                width=s['iheight'],
                depth=s['depth'],
                n_filters=s['nfilters'],
                filter_height=s['fsize'],
                filter_width=s['fsize'],
                img_strides=None,
                filter_strides=None,
                border_mode='valid')
        if 1 < prob_spec.gflops() < 50:
            if prob_spec.is_valid():
                yield prob_spec

def main_step():
    _python, _cmd, dev_id_str, wisdomfile = sys.argv

    device = init_cuda(int(dev_id_str))

    try:
        wdb, results, rng = cPickle.load(open(wisdomfile))
    except (IOError, EOFError):
        wdb, results, rng = wisdom.Wisdom(), [], np.random.RandomState(2)

    while True:
        ii = len(results)

        pgen = problem_generator(rng)
        prob_spec = pgen.next()
        print "=" * 80
        print "PROB GFLOPS", prob_spec.gflops()

        print prob_spec
        wdb.build_dtree(rng=rng, force=True)
        wdb.print_dtree()

        finding = {}
        def add_finding(k, timing):
            op_spec = timing.op_spec
            if op_spec == getattr(finding.get('ref', None), 'op_spec', None):
                print "EXP: REUSING REF", k
                finding[k] = finding['ref']
            else:
                print "EXP: MEASURING ", k, "...",
                test_timing = wisdom.Timing(prob_spec, op_spec)
                test_timing.measure(device)
                if test_timing.valid:
                    print "EXP: MEASURED ", k, test_timing.speed()
                    finding[k] = test_timing
                    finding[k+'_orig'] = timing
                else:
                    print "ERROR: INVALID TEST TIMING", k

        # don't reuse test_timing, it would be cheating
        timing = wisdom.Timing(prob_spec, wisdom.reference_op_spec())
        timing.measure(device)
        if not timing.valid:
            continue
        add_finding('ref', timing)

        if 0:
            rand_timing = timing
            for ii in xrange(75):
                rand_timing = wisdom.genetic_step(
                        rand_timing, device, mutation_rate=1.0, rng=rng)
                if ii == 1 - 1:
                    add_finding('rand1', rand_timing)
                if ii == 25 - 1:
                    add_finding('rand25', rand_timing)
                if ii == 50 - 1:
                    add_finding('rand50', rand_timing)
                if ii == 75 - 1:
                    add_finding('rand75', rand_timing)

        TOPN = 75

        if 0:  # tree-filtered genetic search
            tree_timing = timing
            tree_wisdom = wisdom.Wisdom()
            ref_speed = timing.speed()
            tree_wisdom.record(prob_spec, timing.op_spec, ref_speed,
                    ref_speed)
            for ii in xrange(TOPN):
                tree_timing = wisdom.tree_step(
                        tree_timing, device, rng=rng, wisdom=tree_wisdom,
                        N=3, mutation_rate=.25,
                        ref_speed=ref_speed)
                if ii == 1-1:
                    add_finding('tree1', tree_timing)
                if ii == 25-1:
                    add_finding('tree25', tree_timing)
                if ii == 50-1:
                    add_finding('tree50', tree_timing)
                if ii == 75-1:
                    add_finding('tree75', tree_timing)

        if 0:  # tree-filtered random search (accumulating wisdom)
            wise_timing = timing
            ref_speed = timing.speed()
            wdb.record(prob_spec, timing.op_spec, ref_speed,
                    ref_speed)
            for ii in xrange(TOPN):
                wise_timing = wisdom.tree_step(
                        wise_timing, device, rng=rng, wisdom=wdb,
                        N=3, mutation_rate=.25,
                        ref_speed=ref_speed)
                if ii == 1 - 1:
                    add_finding('wise1', wise_timing)
                if ii == 25 - 1:
                    add_finding('wise25', wise_timing)
                if ii == 50 - 1:
                    add_finding('wise50', wise_timing)
                if ii == 75 - 1:
                    add_finding('wise75', wise_timing)


        if 1:  # genetic search
            gen_timing = timing
            for ii in xrange(TOPN):
                gen_timing = wisdom.genetic_step(
                        gen_timing, device, mutation_rate=0.25, rng=rng)
                if ii == 1 - 1:
                    add_finding('gen1', gen_timing)
                if ii == 25 - 1:
                    add_finding('gen25', gen_timing)
                if ii == 50 - 1:
                    add_finding('gen50', gen_timing)
                if ii == 75 - 1:
                    add_finding('gen75', gen_timing)

        if 1:
            grid_timing = wisdom.gcg_grid_autotune(timing, device)
            add_finding('grid', grid_timing)

        results.append(finding)
        ofile = open(wisdomfile, 'w')
        cPickle.dump((wdb, results, rng), ofile)
        ofile.close()

def main_fig1():
    _python, _cmd, wisdomfile = sys.argv
    wdb, results, rng = cPickle.load(open(wisdomfile))
    import matplotlib.pyplot as plt
    for key, col, marker in [
            ('rand25', (.50, 0, 0), '+'),
            ('rand50', (.75, 0, 0), '+'),
            ('rand75', (.99, 0, 0), '+'),
            ('gen25',  (0, .50, 0), '+'),
            ('gen50',  (0, .75, 0), '+'),
            ('gen75',  (0, .99, 0), '+'),
            ('tree25', (0, 0, .50), '+'),
            ('tree50', (0, 0, .75), '+'),
            ('tree75', (0, 0, .99), '+'),
            ('wise25', (.50, 0, .50), '+'),
            ('wise50', (.75, 0, .75), '+'),
            ('wise75', (.99, 0, .99), '+'),
            ('grid',   (0,  0, 0), '+'),
            ]:
        def getspeed(r, k='ref'):
            if k in r and r[k].valid:
                return r[k].speed()
            else:
                return None
        a = np.asarray([getspeed(r, key) for r in results]).astype('float')
        a_orig = np.asarray([getspeed(r, key + '_orig') for r in results]).astype('float')
        b = np.asarray([getspeed(r) for r in results]).astype('float')
        if np.all(1 - np.isfinite(a)):
            continue
        plt.scatter(np.arange(len(a)),
                a / b + .01 * np.random.rand(),
                label=key, c=col, marker=marker)
        plt.scatter(np.arange(len(a))+.25,
                a_orig / b + .01 * np.random.rand(),
                c=col, marker=marker)
        #gmean = np.exp( np.log(a/b).mean())
        #plt.axhline(gmean, c=col)

    plt.xlabel('random trial')
    plt.ylabel('speed up over reference')
    plt.legend(loc='lower left')
    plt.show()


def main_fig_gflops():
    _python, _cmd, wisdomfile = sys.argv
    wdb, results, rng = cPickle.load(open(wisdomfile))
    import matplotlib.pyplot as plt
    for key, col, marker in [
            ('rand25', (.50, 0, 0), '+'),
            ('rand50', (.75, 0, 0), '+'),
            ('rand75', (.99, 0, 0), '+'),
            ('gen25',  (0, .50, 0), '+'),
            ('gen50',  (0, .75, 0), '+'),
            ('gen75',  (0, .99, 0), '+'),
            ('tree25', (0, 0, .50), '+'),
            ('tree50', (0, 0, .75), '+'),
            ('tree75', (0, 0, .99), '+'),
            ('wise25', (0, .50, .50), '+'),
            ('wise50', (0, .75, .75), '+'),
            ('wise75', (0, .99, .99), '+'),
            ('grid',   (0,  0, 0), '+'),
            ]:
        def getspeed(r, k='ref'):
            if k in r:
                return r[k].speed()
            else:
                return None
        a = np.asarray([getspeed(r, key) for r in results]).astype('float')
        a_orig = np.asarray([getspeed(r, key + '_orig')
            for r in results]).astype('float')
        if 0:
            plt.scatter(np.arange(len(a)).astype('float'),
                    a,
                    label=key, c=col, marker=marker)
        plt.scatter(np.arange(len(a)).astype('float')+.25,
                a_orig,
                #label=key,
                c=col, marker=marker)
        #gmean = np.exp( np.log(a/b).mean())
        #plt.axhline(gmean, c=col)

    plt.xlabel('random trial')
    plt.ylabel('speed')
    plt.legend(loc='lower left')
    plt.show()


def main_fig_time():
    _python, _cmd, wisdomfile = sys.argv
    wdb, results, rng = cPickle.load(open(wisdomfile))
    import matplotlib.pyplot as plt
    for key, col, marker in [
            ('rand25', (.50, 0, 0), '+'),
            ('rand50', (.75, 0, 0), '+'),
            ('rand75', (.99, 0, 0), '+'),
            ('gen25',  (0, .50, 0), '+'),
            ('gen50',  (0, .75, 0), '+'),
            ('gen75',  (0, .99, 0), '+'),
            ('tree25', (0, 0, .50), '+'),
            ('tree50', (0, 0, .75), '+'),
            ('tree75', (0, 0, .99), '+'),
            ('wise25', (0, .50, .50), '+'),
            ('wise50', (0, .75, .75), '+'),
            ('wise75', (0, .99, .99), '+'),
            ('grid',   (0,  0, 0), '+'),
            ]:
        def getspeed(r, k='ref'):
            if k in r:
                return r[k].prob_spec.gflops() / r[k].speed()
            else:
                return None
        a = np.asarray([getspeed(r, key) for r in results]).astype('float')
        a_orig = np.asarray([getspeed(r, key + '_orig')
            for r in results]).astype('float')
        plt.scatter(np.arange(len(a)).astype('float'),
                a,
                label=key, c=col, marker=marker)
        if 0:
            plt.scatter(np.arange(len(a)).astype('float')+.25,
                    a_orig,
                    #label=key,
                    c=col, marker=marker)
        #gmean = np.exp( np.log(a/b).mean())
        #plt.axhline(gmean, c=col)

    plt.xlabel('random trial')
    plt.ylabel('time')
    plt.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':
    cmd = sys.argv[1]
    main = globals()['main_' + cmd]

    sys.exit(main())
