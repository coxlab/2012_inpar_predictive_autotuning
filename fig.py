"""
Scripts for making figures

"""

import cPickle
import logging
import sys
from os import environ

import numpy as np
# -- 
import matplotlib.pyplot as plt
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams['ps.useafm'] = True 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['ps.usedistiller'] = 'xpdf'

PLOT = int(environ.get("PLOT", 0))


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# *try* to convince mpl to do something non-crappy
def adjust_axis(ax):

    for loc, spine in ax.spines.iteritems():
        if loc in ['left', 'bottom']:
            spine.set_position(('outward', 10))
            spine.set_linewidth(0.5)
        if loc in ['right', 'top']:
            spine.set_color('none')

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    fontsize = 8
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_family('sans-serif')
    for line in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        line.set_markeredgewidth(0.5)


def pretty_dname(devicename):
    if devicename == '295': return 'GTX 295'
    if devicename == '480': return 'GTX 480'
    if devicename == '580': return 'GTX 580'
    if devicename == '1060': return 'Tesla C1060'
    if devicename == '2070': return 'Tesla C2070'


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


def main_allstars_mixup():
    """
    Produce a scatterplot of ref vs. mismatched auto-tuned kernels
    """
    _python, _cmd, wisdomfile, timingfile = sys.argv
    what = '580'
    #what = '295'
    assert what in wisdomfile # otw fix save mechanism
    _wdb, results, _rng = cPickle.load(open(wisdomfile))
    timings = cPickle.load(open(timingfile))
    print "LEN TIMINGS", len(timings)
    print "N VALID", len([t for t in timings if t.valid])

    from matplotlib import rc
    rc('text', usetex=True)
    #rc('font', family='serif')

    f = plt.figure()#figsize=(8, 12), dpi=100, facecolor='w')
    ax = f.add_subplot(1,1,1)#, aspect=1.2)
    plt.hold(True)

    ax.hold(True)

    i = 0

    x = []
    y = []
    z = []
    for finding_dct in results:
        if 'gen75' in finding_dct:
            gen75 = finding_dct['gen75']
            if i >= len(timings):
                break
            timing = timings[i]
            x.append(gen75.speed())
            if timing.valid:
                y.append(timing.speed())
            else:
                y.append(0)
            z.append(finding_dct['ref'].speed())
            i += 1
    print x
    print y
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    xm = np.mean(x)
    ym = np.mean(y)
    zm = np.mean(z)
    xe = np.std(x) / np.sqrt(xm.size)
    ye = np.std(y) / np.sqrt(ym.size)
    ze = np.std(z) / np.sqrt(zm.size)
    #gvr = x / z
    #bvr = y / z
    #gvrm = gvr.mean()
    #bvrm = bvr.mean()
    #gvre = gvr.std() / np.sqrt(gvr.size)
    #bvre = bvr.std() / np.sqrt(bvr.size)
    #print gvre, bvre
    plt.bar([0.75, 1.75, 2.75],
            [xm, zm, ym],
            #yerr=[xe, ze, ye],
            width=[0.5, 0.5, 0.5])
    plt.xticks([1, 2, 3],
               [
        'Auto-tuned',
        'Reference',
        'Mismatched Auto-tuned'])
    #plt.bar([0.75, 1.75],#,, 2.75],
            #[gvrm, bvrm],
            #yerr=[gvre, bvre],
            ##[xm, zm, ym],
            ##yerr=[xe, ze, ye],
            #width=[0.5, 0.5])#, 0.5])
    #plt.xticks([1, 2],#, 3],
               #[
        #'Auto-tuned vs. Reference',
        ##'Reference',
        #'Mismatched Auto-tuned vs. Reference'])

    adjust_axis(ax)

    plt.ylabel("Average GFLOPS/s")
    if PLOT:
        plt.show()
    else:
        plt.savefig('fig_allstars_mixup_%s.pdf' % what)


def main_genX():
    _python, _cmd, hostname, devicename = sys.argv
    results = step_timings(hostname, devicename)

    def getel(f):
        if not ('ref' in f and f['ref'].valid): return None
        if not ('gen25' in f and f['gen25'].valid): return None
        if not ('gen50' in f and f['gen50'].valid): return None
        if not ('gen75' in f and f['gen75'].valid): return None
        if not ('grid' in f and f['grid'].valid): return None
        return [f[k].speed() for k in 'ref', 'gen25', 'gen50', 'gen75',
                'grid']
    results = [f for f in results if getel(f) is not None]
    ref, gen25, gen50, gen75, grid = zip(*map(getel, results))

    ref = np.asarray(ref)
    gen25 = np.asarray(gen25)
    gen25 = np.asarray(gen50)
    gen25 = np.asarray(gen75)
    grid = np.asarray(grid)

    axes= plt.axes()
    lines = axes.boxplot(
            x = [gen25/ref, gen50/ref, gen75/ref, grid/ref],
            widths=[0.7] * 4,
            #positions=[0, 1, 2, 3, 4],
            whis=100,
            )
    plt.ylabel('Speedup over reference (%s)' % pretty_dname(devicename))
    plt.xticks([1, 2, 3, 4], ['HC25', 'HC50', 'HC75', 'grid'])
    if devicename in ('480', '580', '2070'):
        axes.set_yscale('log')
        axes.set_ylim(0.5, 33)
        plt.yticks([1, 2, 4, 8, 16, 32],
                [('%ix'%i) for i in [1, 2, 4, 8, 16, 32]])
    else:
        yticks = [0.5, 1, 1.5, 2, 2.5]
        plt.yticks(yticks, [('%.1fx'%i) for i in yticks])

    if PLOT:
        plt.show()
    else:
        figname = 'fig_genX_%s_%s.pdf' % (hostname, devicename)
        print 'writing figure to', figname
        plt.savefig(figname)


def main_gflop_scatter():
    _python, _cmd, hostname, devicename, inv_mult, n_train = sys.argv
    results = step_timings(hostname, devicename)
    mdl_timings = test_timings(hostname, devicename, inv_mult, n_train)

    from matplotlib import rc
    rc('text', usetex=True)

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

    f = plt.figure(figsize=(6, 4), dpi=100, facecolor='w')
    ax = f.add_subplot(1,1,1, aspect=1.2)
    plt.hold(True)

    plt.subplots_adjust(left = 0.0,
                    right = 0.9,
                    bottom = 0.15,
                    top = 0.95,
                    wspace = 0.1,
                    hspace = 0.45)
    ax.hold(True)

    plt.scatter(hc_speeds, tree_speeds, c='b')

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

    adjust_axis(ax)

    plt.xlabel('GFLOP/s of empirical auto-tuning')
    plt.ylabel('GFLOP/s of predictive auto-tuning ')
    #plt.legend(loc='lower left')
    if PLOT:
        plt.show()
    else:
        print 'saving to ', figname
        plt.savefig(figname)

def main_ntrain():
    _python, _cmd, hostname, devicename = sys.argv
    results = step_timings(hostname, devicename)

    def getpspecspeed(pspec, k):
        for r in results:
            if k in r and r[k].prob_spec == pspec:
                if r[k].valid:
                    return r[k].speed()
                else:
                    return None
    def use_this(p, t):
        return t.valid and getpspecspeed(p, 'ref') is not None

    from matplotlib import rc
    rc('text', usetex=True)
    #rc('font', family='serif')

    f = plt.figure()#figsize=(8, 12), dpi=100, facecolor='w')
    ax = f.add_subplot(1,1,1)#, aspect=1.2)
    plt.hold(True)

    ax.hold(True)

    all_hc_speeds = []
    all_ref_speeds = []
    for inv_mult, col in [('0.01', 'r'),
            ('0.25', 'g'),
            ('0.5', 'b'),
            ('0.75', 'c'),
            ('0.99', 'm')
                         ]:
        xx = []
        yy = []
        yy_v = []
        for n_train in '10', '25', '50', '100', '200':
            train_result = test_timings(hostname, devicename, inv_mult,
                    n_train)
            mdl_timings = [(p, t)
                    for (p, t) in train_result.iteritems()
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
        plt.errorbar(xx, yy, yerr=yy_v, c=col, label=r'$\zeta=\log(%s)$' % inv_mult)

    plt.xlim(-10, 220)
    plt.axhline(np.mean(all_hc_speeds), c='k', ls='--')
    plt.text(-5, np.mean(all_hc_speeds)+1.5, 'Auto-tuned mean')
    plt.axhline(np.mean(all_ref_speeds), c='k', ls='--')
    plt.text(-5, np.mean(all_ref_speeds)+1.5, 'Reference mean')

    if '295'  == devicename:
        plt.title('GTX 295')
    elif '480'  == devicename:
        plt.title('GTX 480')
    elif '580'  == devicename:
        plt.title('GTX 580')
    elif '1060'  == devicename:
        plt.title('Tesla C1060')
    elif '2070'  == devicename:
        plt.title('Tesla C2070')
    elif '8600'  == devicename:
        plt.title('8600GT')

    adjust_axis(ax)

    figfile = 'fig_ntrain_%s_%s.pdf' % (hostname, devicename,)
    plt.xlabel('N. auto-tuned problem configurations used for training')
    plt.ylabel('GFLOP/s')
    plt.legend(loc='lower right')
    if PLOT:
        plt.show()
    else:
        print 'saving figure', figfile
        plt.savefig(figfile)


if __name__ == '__main__':
    cmd = sys.argv[1]
    main = globals()['main_' + cmd]

    sys.exit(main())
