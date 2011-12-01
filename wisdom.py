import logging
import time
import fbconv3_cuda
import numpy as np
import scipy as sp
from hyperopt.ht_dist2 import one_of, rSON2
import pycuda.driver

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
RSEED = 123

class CudaContext(object):
    def __init__(self, device):
        self.device = device
    def __enter__(self):
        self.context = self.device.make_context()
        return self.context
    def __exit__(self, *args):
        #print "MEM INFO", pycuda.driver.mem_get_info()
        self.context.pop()
        self.context.detach()


class OpSpec(object):
    def __init__(self,
            block_w,
            block_h,
            n_filter_rows,
            n_output4s,
            spill,
            imul_fast,
            pad_shared,
            use_tex1dfetch,
            maxrregcount,
            use_fast_math,
            ):
        self.__dict__.update(locals())
        del self.self

    def __eq__(self, other):
        return (type(self) == type(other)
                and (self.feature_pairs() == other.feature_pairs()))

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((type(self),) + tuple(self.feature_pairs()))

    def __repr__(self):
        assigns = ['%s=%s' % (n, v) for v, n in self.feature_pairs()]
        return "OpSpec(%s)" % ", ".join(assigns)

    def FilterOp(self, imgs, filters, outs, ctxt):
        return fbconv3_cuda.FilterOp(imgs, filters, outs,
                ctxt=ctxt, **self.__dict__)

    def feature_pairs(self):
        return [(self.block_w, 'block_w'),
                (self.block_h, 'block_h'),
                (self.n_filter_rows, 'n_filter_rows'),
                (self.n_output4s=="all", 'output4s_all'),
                (self.n_output4s==1, 'output4s_1'),
                (self.n_output4s==2, 'output4s_2'),
                (self.spill, 'spill'),
                (self.imul_fast, 'imul_fast'),
                (self.pad_shared, 'pad_shared'),
                (self.use_tex1dfetch, 'tex1dfetch'),
                (self.maxrregcount==None, 'maxreg_none'),
                (0 if self.maxrregcount is None else self.maxrregcount,
                    'maxreg'),
                (self.use_fast_math, 'fast_math'),
                ]
    def feature_names(self):
        return zip(*self.feature_pairs())[1]

    def feature_values(self):
        return map(float, zip(*self.feature_pairs())[0])

def random_op_spec(rng):
    dct = rSON2(
        "block_w" , one_of(4, 8, 16, 32, 64, 128),
        "block_h" , one_of(4, 8, 16, 32, 64, 128),
        "n_filter_rows" , one_of(1, 2),
        "n_output4s" , one_of("all", 1, 2),
        "spill" , one_of(False, True),
        "imul_fast" , one_of(False, True),
        "pad_shared" , one_of(False, True),
        "use_tex1dfetch" , one_of(False, True),
        "maxrregcount" , one_of(None, 8, 16, 20, 24, 28, 32),
        "use_fast_math", one_of(False, True),
        ).sample(rng)
    return OpSpec(**dct)

def random_op_cross(op1, op2, rng, r=.5):
    return OpSpec(
            op1.block_w if rng.rand() < r else op2.block_w,
            op1.block_h if rng.rand() < r else op2.block_h,
            op1.n_filter_rows if rng.rand() < r else op2.n_filter_rows,
            op1.n_output4s if rng.rand() < r else op2.n_output4s,
            op1.spill if rng.rand() < r else op2.spill,
            op1.imul_fast if rng.rand() < r else op2.imul_fast,
            op1.pad_shared if rng.rand() < r else op2.pad_shared,
            op1.use_tex1dfetch if rng.rand() < r else op2.use_tex1dfetch,
            op1.maxrregcount if rng.rand() < r else op2.maxrregcount,
            op1.use_fast_math if rng.rand() < r else op2.use_fast_math,
            )


def reference_op_spec():
    return OpSpec(block_w=8,
            block_h=8,
            n_filter_rows=1,
            n_output4s='all',
            spill=False,
            imul_fast=False,
            pad_shared=True,
            use_tex1dfetch=False,
            maxrregcount=None,
            use_fast_math=False,)


class ProblemSpec(object):
    def __init__(self,
            n_imgs,        # images processed at once
            height,        # image height
            width,         # image width
            depth,         # image depth
            n_filters,     # number of filters
            filter_height,       # filter height
            filter_width,        # filter width
            img_strides,   # how is image physically strided
            filter_strides,# how is filter physically strided
            border_mode,   # one of 'valid', 'full', 'same'
            device=None,
            ):
        self.__dict__.update(locals())
        del self.self
        if self.border_mode == 'valid':
            self.out_height = self.height - self.filter_height + 1
            self.out_width = self.width - self.filter_width + 1
        elif self.border_mode == 'full':
            self.out_height = self.height + self.filter_height - 1
            self.out_width = self.width + self.filter_width - 1
        elif self.border_mode == 'same':
            self.out_height = self.height
            self.out_width = self.width
        else:
            raise ValueError(self.border_mode)

        if n_imgs != 1:
            raise NotImplementedError()
        if border_mode != 'valid':
            raise NotImplementedError()

    def gflops(self):
        return (self.n_imgs
                * self.out_height * self.out_width * self.n_filters
                * self.filter_height * self.filter_width * self.depth
                * 2  # mul and add
                / (1000.**3.)) #return as giga float ops

    # relevant hand-designed features of problem specification
    def feature_pairs(self):
        # all imgs c contiguous
        # each img c contiguous
        # all imgs f contiguous
        # each img f contiguous
        # all imgs size
        # each img size
        # img channel major
        # img channel minor
        # all filters c contiguous
        # each filter c contiguous
        # all filters f contiguous
        # each filter f contiguous
        # all filters size
        # each filter size
        # filter channel major
        # filter channel minor
        # filters flipped vertically (conv vs. corr)
        # filters flipped horizontally (conv vs. corr)
        names = [
            'n_imgs',
            'height',
            'width',
            'depth',
            'n_filters',
            'filter_height',
            'filter_width',
            ]
        return [(getattr(self, name), name) for name in names]

    def feature_names(self):
        return zip(*self.feature_pairs())[1]

    def feature_values(self):
        return map(float, zip(*self.feature_pairs())[0])


    def __repr__(self):
        assigns = ['%s=%s' % (n, v) for v, n in self.feature_pairs()]
        return "ProblemSpec(%s)" % ", ".join(assigns)

    def plan(self, patience=0.0, wisdom=None, approx_n_uses=1000, verbose=0,
            device=None, rng=None):
        """
        problem_spec - ProblemSpec instance
        patience - return a plan within this many seconds.
        wisdom - a Wisdom object (for reading and writing)
        approx_n_uses - estimated number of times this plan will be used (for budgeting search)

        Returns a FilterOp object
        """
        t_start = time.time()
        # -- start by getting something to return ASAP
        encumbent = reference_op_spec()
        if (time.time() - t_start) >= patience:
            return encumbent

        if wisdom is None:
            wisdom = Wisdom()

        encumbent_speed = self.measure_speed(encumbent, n_warmups=2, n_runs=8,
                wisdom=wisdom, device=device)

        n_clocked = [0]
        def clock_candidate():
            n_clocked[0] += 1
            return self.measure_speed(candidate,
                    n_warmups=2,
                    n_runs=8,
                    abort_thresh=encumbent_speed * 0.75,
                    save_on_abort=False,
                    wisdom=wisdom,
                    device=device)

        for candidate in wisdom.ranked_suggestions(self)[:3]:
            if candidate == encumbent:
                continue
            if (time.time() - t_start) >= patience:
                return encumbent
            candidate_speed = clock_candidate()
            if candidate_speed > encumbent_speed:
                encumbent = candidate
                encumbent_speed = candidate_speed

        if rng is None:
            rng = np.random.RandomState(int(time.time() * 1000))

        # XXX: instead of drawing randomly
        #      - draw randomly and filter using the dtree
        #      - randomly perturb and hillclimb from the encumbent
        #      - run some other kind of optimization strategy here
        while (time.time() - t_start) < patience:
            candidate = random_op_cross(encumbent,
                    random_op_spec(rng),
                    rng, .75) # XXX: local search should be parametrized somehow
            candidate_speed = clock_candidate()
            if candidate_speed > 0:
                if candidate_speed > encumbent_speed:
                    encumbent = candidate
                    encumbent_speed = candidate_speed
        #print 'N_CLOCKED', n_clocked[0]
        return encumbent

    def measure_speed(self, op_spec, n_warmups, n_runs, abort_thresh=None,
            wisdom=None, save_on_abort=True, device=None):
        """Return GFLOPS/S of op, not counting transfer times.

        abort_thresh - return 0 if the true value appears to be
                       lower than this (for early exit).
        """
        all_timings = {}
        gflop = self.gflops()

        img_shp = (self.height, self.width, self.depth)
        ker_shp = (self.n_filters, self.filter_height, self.filter_width,
                self.depth)
        out_shp = (self.out_height, self.out_width, self.n_filters)

        sp.random.seed(RSEED)
        in_data = sp.random.randn(*img_shp).astype('float32')
        fb_data = sp.random.randn(*ker_shp).astype( 'float32')
        out_data = sp.empty(out_shp, dtype='float32')

        timings = dict([(key, [])
                        for key in ('upload',
                                    #'set_up',
                                    'process',
                                    'cuda',
                                    'download',
                                    )])

        with CudaContext(device) as context:

            # XXX one image at a time
            in_ = fbconv3_cuda.Input(*img_shp)
            fb_ = fbconv3_cuda.Filterbank(*ker_shp)
            out_ = fbconv3_cuda.Output(*out_shp)

            # -- set-up operation (i.e. compilation)
            fb_[:] = 0
            try:
                fop = op_spec.FilterOp(in_, fb_, out_, ctxt=context)
            except (fbconv3_cuda.InvalidConfig,
                    pycuda._driver.LogicError,    #XXX: cuModuleGetTexRef not found
                    pycuda.driver.CompileError,): #XXX: using too much shared memory
                if wisdom and save_on_abort:
                    wisdom.record(self, op_spec, 0)
                print '-- InvalidConfig'
                return 0

            for i in xrange(n_warmups + n_runs):

                # -- upload data
                start = time.time()
                in_[:] = in_data
                # XXX: is writing a 0 here important for correctness?
                out_[:] = 0
                fb_[:] = fb_data
                end = time.time()
                t_upload = end - start

                # -- process convolution
                # Note Bene: Filter != Conv
                start = time.time()
                try:
                    t_cuda = fop()
                except fbconv3_cuda.InvalidConfig:
                    if wisdom and save_on_abort:
                        wisdom.record(self, op_spec, 0)
                    print '-- InvalidConfig'
                    return 0
                end = time.time()
                t_process = end - start

                if i > 0 and (gflop / t_cuda < abort_thresh):
                    if wisdom and save_on_abort:
                        wisdom.record(self, op_spec, gflop / t_cuda)
                    print '-- InvalidConfig'
                    return 0

                start = time.time()
                out_data = out_[:]
                end = time.time()
                t_download = end - start

                if i >= n_warmups:
                    timings['upload'] += [t_upload]
                    timings['process'] += [t_process]
                    timings['cuda'] += [t_cuda]
                    timings['download'] += [t_download]

                if 0:
                    gflops_cuda = gflop / t_cuda
                    gflops_proc = gflop / t_process
                    gflops_tot = gflop / (t_process+t_upload+t_download)
                    print "gflops_cuda", gflops_cuda
                    print "gflops_proc", gflops_proc
                    print "gflops_tot", gflops_tot


            timings_stats = dict([(key, {'median': sp.median(t),
                                         'mean': sp.mean(t),
                                         'std': sp.std(t),
                                         'max': max(t),
                                         'min': min(t),
                                         }
                                   )
                                  for key, t in timings.iteritems()
                                  ]
                                 )

            gflops_cuda = {
                'median': gflop / timings_stats['cuda']['median'],
                'mean': gflop / timings_stats['cuda']['mean'],
                'max': gflop / timings_stats['cuda']['min'],
                'min': gflop / timings_stats['cuda']['max'],
                }

            print "GFLOPS_CUDA", gflops_cuda

            if wisdom:
                wisdom.record(self, op_spec, gflops_cuda['max'])
            return gflops_cuda['max']

    def assert_correct(self, op):
        """raise assertion error if op() produces incorrect output
        """
        raise NotImplementedError()


class Wisdom(object):
    """
    Wisdom takes the form of a set of good kernels and a decision function for
    choosing between them.
    """

    def __init__(self):
        self._observations = []
        self._dtree_n_obs = 0
        self._dtree = None

    def _suggest_helper(self, feature, node, prefix=""):
        if node['kind'] == 'fork':
            print prefix,
            if node['feature'] < len(feature):
                if feature[node['feature']] < node['value']:
                    print '-- branch ', node['feature_name'], '<', node['value']
                    child = node['left']
                else:
                    print '-- branch ', node['feature_name'], '>=', node['value']
                    child = node['right']
                return self._suggest_helper(feature, child, prefix+"  ")
            else:
                print '-- ignoring', node['feature_name'], '<', node['value']
                lval = self._suggest_helper(feature, node['left'], prefix+"  ")
                rval = self._suggest_helper(feature, node['right'], prefix+"  ")
                return lval + rval
        else:
            if node['mean'] < -20: # these runs are failures
                return []
            else:
                return [(node['mean'], node['idxs'])]


    def ranked_suggestions(self, prob_spec):
        if self._dtree is None:
            return [reference_op_spec()]

        # 1. loop over all leaf nodes of self._dtree
        # 2. if leaf is consistent with prob_spec, then
        #    add the best (speed, op) from the leaf set
        # 3. sort all matching pairs by decreasing speed
        #    and return the ops of that list

        scores_idxs = self._suggest_helper(
                np.asarray(prob_spec.feature_values()),
                self._dtree)
        rval_specs = set()
        rvals = []
        for score, idxs in scores_idxs:
            for ii in idxs:
                op_spec = self._observations[ii][1]
                op_speed = self._observations[ii][2]
                if op_spec not in rval_specs:
                    #XXX if an op_spec appears multiple times scoring various
                    #    speeds for various problems, which score should we
                    #    report here?
                    rval_specs.add(op_spec)
                    rvals.append((op_speed, op_spec))
        if len(rvals) == 0:
            return [reference_op_spec()]
        else:
            rvals.sort()
            rvals.reverse()
            if 0:
                print 'compatible', len(scores_idxs)
                for r in rvals[:5]:
                    print 'RANKED SUG', r
            return [r[1] for r in rvals]

    def record(self, prob_spec, op_spec, speed):
        for ii, (pspec, ospec, s) in enumerate(self._observations):
            if (pspec == prob_spec and ospec == op_spec):
                if abs(np.log(s) - np.log(speed)) > np.log(1.2):
                    logger.error('duplicate entry w different speed: %s, %s' % (
                        s, speed))
                    self._observations[ii]= (
                            pspec, ospec, .5 * s + .5 * speed)
                    return
                    #raise Exception('duplicate entry w different speed',
                    #        (s, speed))
                else:
                    logger.debug('ignoring duplicate entry: %s, %s' % (
                        prob_spec, op_spec))
                    return
        #print "RECORD", (prob_spec, op_spec, speed)
        self._observations.append((prob_spec, op_spec, speed))

    def build_dtree_rec(self, features, targets, global_idxs, feature_names,
            min_improvement=.1,
            min_split_size=10):
        assert len(features) == len(targets) == len(global_idxs)
        assert features.shape[1] == len(feature_names)
        targets_var = np.var(targets)
        total_sse = (len(targets) * targets_var)
        logger.debug('total squared error = %f' % total_sse)
        if total_sse < min_improvement:
            return dict(
                    kind='leaf',
                    mean=np.mean(targets),
                    var=targets_var,
                    idxs=global_idxs)

        best_sse = float('inf')

        for i in xrange(features.shape[1]):
            features_i = features[:, i]
            order_i = np.argsort(features_i)

            sorted_target = targets[order_i]

            # XXX : do this in linear time instead of quadratic!
            # XXX: check for off by 1
            for j in xrange(min_split_size, len(features) - min_split_size):
                if features_i[order_i[j - 1]] != features_i[order_i[j]]:
                    below = sorted_target[:j]
                    above = sorted_target[j:]
                    new_total_sse = (len(below) * np.var(below)
                            + len(above) * np.var(above))
                    if new_total_sse < best_sse:
                        split_pt = (0.5 * features_i[order_i[j - 1]]
                                + 0.5 * features_i[order_i[j]])
                        logger.debug('new best sse %f  (%i, %i) (%s < %s)' % (
                            new_total_sse,
                            i, j,
                            feature_names[i], split_pt))
                        best_ij = (i, j, split_pt)
                        best_sse = new_total_sse

        if best_sse < (total_sse - min_improvement):
            ii, jj, split_pt = best_ij
            one_to_n = np.arange(len(features))
            leftidxs = one_to_n[features[:, ii] < split_pt]
            rightidxs = one_to_n[features[:, ii] >= split_pt]
            assert len(leftidxs) + len(rightidxs) == len(features)
            assert len(leftidxs) >= min_split_size
            assert len(rightidxs) >= min_split_size
            return dict(
                    kind='fork',
                    feature=ii,
                    feature_name=feature_names[ii],
                    value=split_pt,
                    left=self.build_dtree_rec(
                        features[leftidxs],
                        targets[leftidxs],
                        global_idxs[leftidxs],
                        feature_names,
                        ),
                    right=self.build_dtree_rec(
                        features[rightidxs],
                        targets[rightidxs],
                        global_idxs[rightidxs],
                        feature_names,
                        ),
                    )
        else:
            return dict(
                    kind='leaf',
                    mean=np.mean(targets),
                    var=targets_var,
                    idxs=global_idxs)

    def build_dtree(self, force=True):
        if not force:
            if (len(self._observations) < 10 or
                    len(self._observations) < 1.1 * self._dtree_n_obs):
                return
        features = []
        targets = []
        logger.info('building dtree from %i observations' %
                len(self._observations))
        for prob_spec, op_spec, speed in self._observations:
            feature = np.concatenate([
                prob_spec.feature_values(),
                op_spec.feature_values()])
            target = np.log(speed + 1e-10)
            features.append(feature)
            targets.append(target)

        # just use last prob_spec
        feature_names = prob_spec.feature_names() + op_spec.feature_names()
        features = np.asarray(features, order='F')
        targets = np.asarray(targets)

        self._dtree = self.build_dtree_rec(features, targets,
                global_idxs=np.arange(len(features)),
                feature_names=feature_names)
        self._dtree_n_obs = len(features)

        if 0:
            for i, o in enumerate(self._observations):
                print 'OBS', i, o[0]
                print 'OBS', i, o[1]
                print 'OBS', i, o[2]

    def print_dtree(self, node=None, prefix=""):
        if node is None:
            node = self._dtree
        if node is self._dtree:
            print 'DTREE (n_obs = %i)' % len(self._observations)
        if node['kind'] == 'fork':
            print prefix,
            print node['feature_name'], '<', node['value']
            self.print_dtree(node['left'], prefix + "  ")
            self.print_dtree(node['right'], prefix + "  ")
        else:
            print prefix,
            print 'avg of ', len(node['idxs']),
            print ': ', node['mean'],
            print '+-', np.sqrt(node['var'])

