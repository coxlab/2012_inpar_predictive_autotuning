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


def build_logspeed_dataset(wisdomfile, invalid_multiplier, n_train):
    timings = {}
    refspeed = {}

    print "Loading data from ", wisdomfile
    wdb, results, rng = cPickle.load(open(wisdomfile))
    for ii, finding in enumerate(results):
        ref = finding['ref']
        ref_orig = finding['ref_orig']
        key_ii = (ref.prob_spec, wisdomfile)

        refspeed.setdefault(key_ii,
                np.mean([ref.speed(), ref_orig.speed()]))
        timings.setdefault((ref.prob_spec, wisdomfile), [])

        for k, t in finding.iteritems():
            if not k.startswith('ref'):
                t.src = dict(srcfile=wisdomfile, pos=ii, key=k)
                timings[key_ii].append(t)

    print "Out of %i results" % len(results)
    print "Loaded timings for %i problem specs" % len(timings)

    assert n_train + 100 < len(timings)
    train_keys = timings.keys()[:n_train]
    test_keys = timings.keys()[-100:]

    def to_xy(keys):
        x = []
        y = []
        prob_specs = []
        refspeeds = []
        for k in keys:
            for t in timings[k]:
                x.append(t.prob_spec.feature_values() + t.op_spec.feature_values())
                if t.valid:
                    y.append(np.log(t.speed() / refspeed[k]))
                else:
                    y.append(np.log(invalid_multiplier))
                prob_specs.append(t.prob_spec)
                refspeeds.append(refspeed[k])
        return x, y, prob_specs, refspeeds

    train_data = to_xy(train_keys)
    test_data = to_xy(test_keys)

    return train_data, test_data


def main_train():
    _python, _cmd, wisdomfile, dev_id_str, inv_mult, n_train = sys.argv
    train_data, test_data = build_logspeed_dataset(wisdomfile,
            float(inv_mult), int(n_train))
    train_x, train_y, train_pspec, train_refspeeds = train_data
    test_x, test_y, test_pspec, test_refspeeds = test_data

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

    LSMs = []

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
        LSMs.append(np.log(timing.speed() / test_refspeeds[idxs[0]]))
        assert np.var(test_refspeeds[idxs]) < 1e-8, test_refspeeds[idxs]
        print 'Tree: %3.3f  Ref: %3.3f  lsm: %.3f  best lsm %.3f  running: %.3f (%s, %s, %i)' % (
                timing.speed(),
                test_refspeeds[idxs[0]],
                np.log(timing.speed() / test_refspeeds[idxs[0]]),
                np.max(test_y[idxs]),
                np.exp(np.mean(LSMs)),
                inv_mult, n_train, len(LSMs)
                )
    cPickle.dump(mdl_timings, open('train_results_%s_%s_%s' % (wisdomfile, inv_mult, n_train), 'w'))


def main_figtrain():
    _python, _cmd, wisdomfile, inv_mult = sys.argv
    train_result = 'train_results_%s_%s' % (wisdomfile, inv_mult)
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


def main_fig_gflop_scatter():
    _python, _cmd, wisdomfile, inv_mult, n_train = sys.argv
    import matplotlib.pyplot as plt
    train_result = 'train_results_%s_%s_%s' % (wisdomfile, inv_mult, n_train)
    wdb, results, rng = cPickle.load(open(wisdomfile))
    mdl_timings = cPickle.load(open(train_result))

    def getpspecspeed(pspec, k):
        for r in results:
            if k in r and r[k].prob_spec == pspec:
                if r[k].valid:
                    return r[k].speed()
                else:
                    return None
    def use_this(p, t):
        return t.valid and getpspecspeed(p, 'ref') is not None
    hc_speeds = [getpspecspeed(p, 'gen75') for p, t in mdl_timings.items() if use_this(p, t)]
    tree_speeds = [t.speed() for p, t in mdl_timings.items()]
    grid_speeds = [getpspecspeed(p, 'grid') for p, t in mdl_timings.items() if use_this(p, t)]
    ref_speeds = [getpspecspeed(p, 'ref') for p, t in mdl_timings.items() if use_this(p, t)]

    plt.scatter(hc_speeds, tree_speeds, c='b')
    #plt.scatter(ref_speeds, tree_speeds, c='r')

    if '295' in wisdomfile:
        ubound =  350
        plt.title('GTX 295')
        figname = 'fig_gflop_scatter_295.pdf'
    elif '1060' in wisdomfile:
        ubound =  350
        plt.title('Tesla C1060')
        figname = 'fig_gflop_scatter_1060.pdf'
    elif '480' in wisdomfile:
        ubound =  850
        plt.title('GTX 480')
        figname = 'fig_gflop_scatter_480.pdf'
    elif '8600' in wisdomfile:
        ubound =  100
        plt.title('8600GT')
        figname = 'fig_gflop_scatter_8600.pdf'

    plt.plot([0, ubound], [0, ubound], c='k', ls='-')
    plt.plot([0, ubound], [0, ubound / 2.0], c='k', ls='--')
    plt.plot([0, ubound/2.0], [0, ubound], c='k', ls='--')
    plt.text(ubound, ubound, 'equality')
    plt.text(ubound, ubound/2.0, '2x slower')
    plt.text(ubound/2.0, ubound, '2x faster')

    plt.xlabel('GFLOP/s of empirical auto-tuning')
    plt.ylabel('GFLOP/s of predictive auto-tuning ')
    #plt.legend(loc='lower left')
    if 0:
        plt.show()
    else:
        plt.savefig(figname)


def main_fig_ntrain():
    _python, _cmd, wisdomfile = sys.argv
    import matplotlib.pyplot as plt
    wdb, results, rng = cPickle.load(open(wisdomfile))
    def getpspecspeed(pspec, k):
        for r in results:
            if k in r and r[k].prob_spec == pspec:
                if r[k].valid:
                    return r[k].speed()
                else:
                    return None
    def use_this(p, t):
        return t.valid and getpspecspeed(p, 'ref') is not None

    all_hc_speeds = []
    all_ref_speeds = []
    for inv_mult, col in [('0.01', 'r'),
            ('0.25', 'g'),
            ('0.5', 'b'),
            ('0.75', 'c'),
            ('0.99', 'm')]:
        xx = []
        yy = []
        yy_v = []
        for n_train in '10', '25', '50', '100', '200':
            train_result = 'train_results_%s_%s_%s' % (wisdomfile, inv_mult, n_train)
            mdl_timings = [(p, t)
                    for (p, t) in cPickle.load(open(train_result)).iteritems()
                    if use_this(p, t)]
            tree_speeds = [t.speed() for p, t in mdl_timings]
            hc_speeds = [getpspecspeed(p, 'gen75') for p, t in mdl_timings]
            ref_speeds = [getpspecspeed(p, 'ref') for p, t in mdl_timings]
            # scaling is done to space out the error bars
            xx.append(int(n_train) + float(inv_mult)*6)
            yy.append(np.mean(tree_speeds))
            yy_v.append(1.96 * np.std(tree_speeds) / np.sqrt(len(mdl_timings)))
            all_hc_speeds.extend(hc_speeds)
            all_ref_speeds.extend(ref_speeds)
            print inv_mult, n_train, len(mdl_timings), yy[-1]
        plt.errorbar(xx, yy, yerr=yy_v, c=col, label='a=%s' % inv_mult)

    plt.xlim(-10, 220)
    plt.axhline(np.mean(all_hc_speeds), c='k', ls='--')
    plt.text(-5, np.mean(all_hc_speeds)+1.5, 'Auto-tuned mean')
    plt.axhline(np.mean(all_ref_speeds), c='k', ls='--')
    plt.text(-5, np.mean(all_ref_speeds)+1.5, 'Reference mean')

    if '295' in wisdomfile:
        plt.title('GTX 295')
        figfile = 'fig_ntrain_295.pdf'
    elif '480' in wisdomfile:
        plt.title('GTX 480')
        figfile = 'fig_ntrain_480.pdf'
    elif '1060' in wisdomfile:
        plt.title('Tesla C1060')
        figfile = 'fig_ntrain_1060.pdf'
    elif '8600' in wisdomfile:
        plt.title('8600GT')
        figfile = 'fig_ntrain_8600.pdf'
    plt.xlabel('N. auto-tuned problem configurations used for training')
    plt.ylabel('GFLOP/s')
    plt.legend(loc='lower right')
    if 0:
        plt.show()
    else:
        plt.savefig(figfile)


def main_fig_gflop_scatter_theano():
    #
    # This script, if it runs on the 295 shows that theano is about 1/3 the
    # speed on average. On the 480 Theano is more like 1/5 the speed.
    #
    _python, _cmd, wisdomfile, Nstr = sys.argv
    import matplotlib.pyplot as plt
    import bandit
    wdb, results, rng = cPickle.load(open(wisdomfile))

    ref_speeds = []
    gen75_speeds = []
    theano_speeds = []
    grid_speeds = []

    for ii, result in enumerate(results):
        if len(ref_speeds) == int(Nstr):
            break
        if 'ref' in result and result['ref']:
            ref_speed = result['ref'].speed()
        else:
            break
        if 'gen75' in result and result['gen75']:
            gen75_speed = result['gen75'].speed()
        else:
            break
        if 'grid' in result and result['grid']:
            grid_speed = result['grid'].speed()
        else:
            break
        prob_spec = result['ref'].prob_spec
        if prob_spec.height > 512:
            continue
        if prob_spec.n_filters > 256:
            continue
        if prob_spec.depth > 256:
            continue
        foo = bandit.FBCorr3Bandit(
                prob_spec.n_imgs,
                prob_spec.height,
                prob_spec.width,
                prob_spec.depth,
                prob_spec.n_filters,
                prob_spec.filter_height,
                )
        theano_speed = foo.vs_theano()
        print '-' * 80
        print prob_spec
        print prob_spec.gflops()
        print ref_speed, gen75_speed, theano_speed

        ref_speeds.append(ref_speed)
        gen75_speeds.append(gen75_speed)
        grid_speeds.append(grid_speed)
        theano_speeds.append(theano_speed)

    plt.scatter(ref_speeds, gen75_speeds, c='b')
    plt.scatter(ref_speeds, grid_speeds, c='r')
    plt.scatter(ref_speeds, theano_speeds, c='g')

    if 1:
        plt.show()
    else:
        plt.savefig(figname)

def main_theano_conv_slow_so_sad():
    import bandit
    prob_spec = wisdom.ProblemSpec(
            n_imgs=1,
            height=32 ,
            width=32 ,
            depth=32,
            n_filters=64,
            filter_height=5,
            filter_width=5,
            img_strides=None,
            filter_strides=None,
            border_mode='valid')
    prob_spec.n_imgs = 64
    print prob_spec.gflops()
    foo = bandit.FBCorr3Bandit(
            prob_spec.n_imgs,
            prob_spec.height,
            prob_spec.width,
            prob_spec.depth,
            prob_spec.n_filters,
            prob_spec.filter_height,
            )
    print foo.vs_theano()




if __name__ == '__main__':
    cmd = sys.argv[1]
    main = globals()['main_' + cmd]

    sys.exit(main())
