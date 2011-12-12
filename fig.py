"""
Scripts for making figures

"""

import cPickle
import logging
import sys
import time

import numpy as np

import wisdom
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def step_timings(hostname, devicename):
    """
    Retrieve the results list produced by experiment.main_step
    """
    filename = ('timings/%(hostname)s/timing1_%(devicename)s_big.pkl' %
            locals())
    wdb, results, rng = cPickle.load(open(filename))
    return results

def test_timings(hostname, devicename, inv_mult, n_train):
    """
    Retrieve the results of predictive auto-tuning from experiment.main_train
    """
    filename = ('timings/%(hostname)s/train_results_timing1_%(devicename)s_big.pkl_%(inv_mult)s_%(n_train)s' % locals())
    return cPickle.load(open(filename))



def main_gflop_scatter():
    _python, _cmd, hostname, devicename, inv_mult, n_train = sys.argv
    results = step_timings(hostname, devicename)
    mdl_timings = test_timings(hostname, devicename, inv_mult, n_train)

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

    if devicename == '295':
        ubound =  350
        plt.title('GTX 295')
    elif '1060'  == devicename:
        ubound =  350
        plt.title('Tesla C1060')
    elif '2070'  == devicename:
        ubound =  500
        plt.title('Tesla C2070')
    elif '480'  == devicename:
        ubound =  850
        plt.title('GTX 480')
    elif '580'  == devicename:
        ubound =  1200
        plt.title('GTX 580')
    elif '8600'  == devicename:
        ubound =  100
        plt.title('8600GT')
    figname = 'fig_gflop_scatter_%s_%s_%s_%s.pdf' % (
            hostname,
            devicename,
            inv_mult,
            n_train)

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
        print 'saving to ', figname
        plt.savefig(figname)


def main_ntrain():
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

    if '295'  == devicename:
        plt.title('GTX 295')
        figfile = 'fig_ntrain_295.pdf'
    elif '480'  == devicename:
        plt.title('GTX 480')
        figfile = 'fig_ntrain_480.pdf'
    elif '1060'  == devicename:
        plt.title('Tesla C1060')
        figfile = 'fig_ntrain_1060.pdf'
    elif '8600'  == devicename:
        plt.title('8600GT')
        figfile = 'fig_ntrain_8600.pdf'
    plt.xlabel('N. auto-tuned problem configurations used for training')
    plt.ylabel('GFLOP/s')
    plt.legend(loc='lower right')
    if 0:
        plt.show()
    else:
        plt.savefig(figfile)


if __name__ == '__main__':
    cmd = sys.argv[1]
    main = globals()['main_' + cmd]

    sys.exit(main())
