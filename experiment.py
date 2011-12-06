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

def main_count_problem_space_size():
    n_valid = 0
    for iheight in (256, 512, 1024, 2048, 4096):
        for depth in (1, 4, 8, 16, 32, 64, 128, 256):
            for n_filters in (1, 4, 8, 16, 32, 64, 128, 256):
                for fsize in (3, 5, 7, 9, 11):
                    prob_spec = wisdom.ProblemSpec(
                            n_imgs=1,
                            height=iheight,
                            width=iheight,
                            depth=depth,
                            n_filters=n_filters,
                            filter_height=fsize,
                            filter_width=fsize,
                            img_strides=None,
                            filter_strides=None,
                            border_mode='valid')
                    if 1 < prob_spec.gflops() < 50:
                        if prob_spec.is_valid():
                            n_valid += 1
    print n_valid


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
                        gen_timing, device, mutation_rate=0.25, rng=rng,
                        finding=finding)
                if ii == 1 - 1:
                    add_finding('gen1', gen_timing)
                if ii == 25 - 1:
                    add_finding('gen25', gen_timing)
                if ii == 50 - 1:
                    add_finding('gen50', gen_timing)
                if ii == 75 - 1:
                    add_finding('gen75', gen_timing)

        if 1:
            grid_timing = wisdom.gcg_grid_autotune(timing, device, finding)
            add_finding('grid', grid_timing)

        results.append(finding)
        ofile = open(wisdomfile, 'w')
        cPickle.dump((wdb, results, rng), ofile)
        ofile.close()


def main_dump_logspeedup_dataset():
    timings = {}
    refspeed = {}
    try:
        ofile = open(sys.argv[2])
        print 'Please remove output file manually:', sys.argv[2]
        return
    except IOError:
        pass
    for pklfile in sys.argv[3:]:
        assert len(sys.argv) == 4  # XXX: platformfeatures are not implemented
        print "Loading data from ", pklfile
        wdb, results, rng = cPickle.load(open(pklfile))
        for ii, finding in enumerate(results):
            ref = finding['ref']
            ref_orig = finding['ref_orig']
            key_ii = (ref.prob_spec, pklfile)

            refspeed.setdefault(key_ii,
                    np.mean([ref.speed(), ref_orig.speed()]))
            timings.setdefault((ref.prob_spec, pklfile), [])

            for k, t in finding.iteritems():
                if not t.valid:
                    continue
                if not k.startswith('ref'):
                    t.src = dict(srcfile=pklfile, pos=ii, key=k)
                    timings[key_ii].append(t)

        print "Out of %i results" % len(results)
        print "Loaded timings for %i problem specs" % len(timings)

    n_keys = len(timings)
    train_keys = timings.keys()[:int(.75 * n_keys)]
    test_keys = timings.keys()[int(.75 * n_keys):]

    def to_xy(keys):
        x = []
        y = []
        prob_specs = []
        refspeeds = []
        for k in keys:
            for t in timings[k]:
                x.append(t.prob_spec.feature_values() + t.op_spec.feature_values())
                y.append(np.log(t.speed() / refspeed[k]))
                prob_specs.append(t.prob_spec)
                refspeeds.append(refspeed[k])
        return x, y, prob_specs, refspeeds

    train_data = to_xy(train_keys)
    test_data = to_xy(test_keys)

    ofile = open(sys.argv[2], 'w')
    cPickle.dump(train_data, ofile)
    cPickle.dump(test_data, ofile)


def main_train():
    _python, _cmd, dsetfile, dev_id_str = sys.argv
    ifile = open(dsetfile)
    train_x, train_y, train_pspec, train_refspeeds = cPickle.load(ifile)
    test_x, test_y, test_pspec, test_refspeeds = cPickle.load(ifile)
    ifile.close()

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)
    test_refspeeds = np.asarray(test_refspeeds)

    pspec_idxs = {}
    for i, pspec in enumerate(test_pspec):
        pspec_idxs.setdefault(pspec, []).append(i)
    too_short = [pspec for pspec, idxs in pspec_idxs.iteritems()
            if len(idxs) < 10]
    print "deleting", len(too_short), "test pspecs"
    for pspec in too_short:
        del pspec_idxs[pspec]

    print "Loaded", len(train_x), "training examples"
    print "Loaded", len(test_x), "testing examples",
    print "with ", len(pspec_idxs), "sufficiently populated pspecs"

    import sklearn.tree
    import scipy.stats

    print "VAR: ", np.var(test_y)
    cur_train_y = train_y
    cur_pred_y = np.zeros_like(test_y)
    mdls = []
    weak_depth = 4
    for boosting_iter in range(100):
        mdl = sklearn.tree.DecisionTreeRegressor(
                max_depth=weak_depth,
                min_split=10,
                )
        mdl.fit(train_x, cur_train_y)
        cur_train_y -= mdl.predict(train_x)
        cur_pred_y += mdl.predict(test_x)
        rcorr = [scipy.stats.spearmanr(cur_pred_y[idxs], test_y[idxs])[0]
            for pspec, idxs in pspec_idxs.items()]
        #print rcorr
        print "%i: TRAINMSE: %3.3f TESTMSE: %3.3f MRC: %3.3f" % (
                boosting_iter,
                np.mean(cur_train_y ** 2),
                np.mean((cur_pred_y - test_y) ** 2),
                np.mean(rcorr))
        mdls.append(mdl)

    def predict_one(x):
        return np.sum([mdl.predict(x) for mdl in mdls])

    print "Validating Model against actual hardware"
    device = init_cuda(int(dev_id_str))
    rng = np.random.RandomState(123)

    mdl_timings = {}

    for pspec, idxs in pspec_idxs.iteritems():
        t0 = time.time()
        pfeatures = pspec.feature_values()
        op_specs = [wisdom.reference_op_spec()]
        xii = pfeatures + op_specs[-1].feature_values()
        best_score = mdl.predict(np.asarray(xii))
        for ii in xrange(75):
            op_candidate = wisdom.resample_some_coords(
                    op_specs[-1], rng, rate=.25)
            xii = pfeatures + op_candidate.feature_values()
            score = predict_one(np.asarray(xii))
            if score > best_score:
                op_specs.append(op_candidate)
                best_score = score
        print 'search took', time.time() - t0, len(op_specs)
        t1 = time.time()
        timing = None
        while timing is None:
            timing = wisdom.Timing(pspec, op_specs.pop())
            timing.measure(device)
            if not timing.valid:
                timing = None
        print 'backtrack took', time.time() - t1, len(op_specs)

        mdl_timings[pspec] = timing
        assert np.var(test_refspeeds[idxs]) < 1e-8, test_refspeeds[idxs]
        print 'Tree: %3.3f  Ref: %3.3f  lsm: %.3f  best lsm %.3f' % (
                timing.speed(),
                test_refspeeds[idxs[0]],
                np.log(timing.speed() / test_refspeeds[idxs[0]]),
                np.max(test_y[idxs]))
    cPickle.dump(mdl_timings, open(sys.argv[2]+'.train_results', 'w'))


def main_figtrain():
    _python, _cmd, wisdomfile, train_result = sys.argv
    wdb, results, rng = cPickle.load(open(wisdomfile))
    mdl_timings = cPickle.load(open(train_result))
    def getpspecspeed(pspec, k):
        for r in results:
            if k in r and r[k].prob_spec == pspec:
                if r[k].valid:
                    return r[k].speed()
                else:
                    return None
    import matplotlib.pyplot as plt
    print "n. timings: %i   n. valid timings: %i" % (
            len(mdl_timings),
            len([t for p, t in mdl_timings.iteritems() if t.valid]))
    if '295' in train_result:
        logscale_x = False
    elif '480' in train_result:
        logscale_x = True
    elif '1060' in train_result:
        logscale_x = False

    def maybe_log(x):
        if logscale_x:
            return np.log(x)
        else:
            return x
    def use_this(p, t):
        return t.valid and getpspecspeed(p, 'ref') is not None

    print 'Tree Avg GFLOP/s', np.mean([t.speed() for p, t in mdl_timings.items()])
    print 'HC Avg GFLOP/s', np.mean([getpspecspeed(p, 'gen75') for p, t in mdl_timings.items() if use_this(p, t)])
    print 'Grid Avg GFLOP/s', np.mean([getpspecspeed(p, 'grid') for p, t in mdl_timings.items() if use_this(p, t)])
    print 'Ref Avg GFLOP/s', np.mean([getpspecspeed(p, 'ref') for p, t in mdl_timings.items() if use_this(p, t)])

    tree_data = [maybe_log(t.speed() / getpspecspeed(p, 'ref'))
        for p, t in mdl_timings.iteritems() if use_this(p, t)]
    hc_data = [maybe_log(getpspecspeed(p, 'gen75') / getpspecspeed(p, 'ref'))
        for p, t in mdl_timings.iteritems() if use_this(p, t)]
    grid_data = [maybe_log(getpspecspeed(p, 'grid') / getpspecspeed(p, 'ref'))
        for p, t in mdl_timings.iteritems() if use_this(p, t)]

    plt.hist([tree_data, hc_data, grid_data],
        label=[
            'Hill Climbing (model)',
            'Hill Climbing (real)',
            'Grid (real)'],
        color=['b', 'g', 'r'],
        bins=20,
        )

    def vline(val, col, xloc, yloc):
        plt.axvline(val, c=col, ls='--', lw=2)
        plt.annotate('%1.2f' % (np.exp(val) if logscale_x else val),
                xy=(val, yloc),
                xytext=(xloc, yloc),
                arrowprops=dict(
                    width=1,
                    frac=.3,
                    headwidth=5,
                    shrink=.05,
                    color='k',
                    ))
    if '295' in train_result:
        if logscale_x:
            vline(np.mean(tree_data), 'b', 1.2, 20)
            vline(np.mean(hc_data), 'g', 1.55, 21)
            vline(np.mean(grid_data), 'r', 1.55, 19)
        else:
            vline(np.exp(np.mean(np.log(tree_data))), 'b', 1.2, 17)
            vline(np.exp(np.mean(np.log(hc_data))), 'g', 1.55, 18)
            vline(np.exp(np.mean(np.log(grid_data))), 'r', 1.55, 16)
        plt.xlabel('Speed multiplier over reference kernel (GTX 295)')
        savefig_filename = 'speedup_295.pdf'
    if '1060' in train_result:
        if logscale_x:
            vline(np.mean(tree_data), 'b', 1.2, 20)
            vline(np.mean(hc_data), 'g', 1.55, 21)
            vline(np.mean(grid_data), 'r', 1.55, 19)
        else:
            vline(np.exp(np.mean(np.log(tree_data))), 'b', 1.2, 17)
            vline(np.exp(np.mean(np.log(hc_data))), 'g', 1.55, 18)
            vline(np.exp(np.mean(np.log(grid_data))), 'r', 1.55, 16)
        plt.xlabel('Speed multiplier over reference kernel (Tesla C1060)')
        savefig_filename = 'speedup_1060.pdf'
    elif '480' in train_result:
        assert logscale_x
        vline(np.mean(tree_data), 'b', 0.75, 23)
        vline(np.mean(hc_data),   'g', 0.75, 21)
        vline(np.mean(grid_data), 'r', 0.75, 19)
        newticks = np.concatenate([np.arange(1, 10),
                    10 + 10 * np.arange(0, 2)])
        newlabels = [str(i) for i in newticks]
        plt.xticks( np.log(newticks), newlabels )
        plt.xlabel('Speed multiplier over reference kernel (GTX 480)')
        savefig_filename = 'speedup_480.pdf'

    plt.legend(loc='upper right')
    plt.ylabel('Number of test problems')
    if 1:
        plt.show()
    else:
        plt.savefig(savefig_filename)


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
        gmean = np.exp( np.log(a/(b+1e-4)).mean())
        plt.axhline(gmean, c=col)

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
