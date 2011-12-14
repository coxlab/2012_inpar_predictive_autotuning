import sys
import logging
import time
import fbconv3_cuda
import numpy as np
import scipy
try:
    from hyperopt.ht_dist2 import one_of, rSON2
except ImportError:
    print "could not import hyperopt"
    pass
import pycuda.driver

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
        "n_filter_rows" , one_of(1, 2), #XXX: turns out this can be "all" as well
        "n_output4s" , one_of("all", 1, 2),
        "spill" , one_of(False, True),
        "imul_fast" , one_of(False, True),
        "pad_shared" , one_of(False, True),
        "use_tex1dfetch" , one_of(False, True),
        "maxrregcount" , one_of(None, 8, 16, 20, 24, 28, 32),
        "use_fast_math", one_of(False, True),
        ).sample(rng)
    return OpSpec(**dct)

# XXX: n_output4s should offer more choices!
# XXX: maxrregcount should go higher: 64? 128?

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

def resample_some_coords(op, rng, rate=.5):
    rop = random_op_spec(rng)
    return random_op_cross(rop, op, rng, rate)


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

        if img_strides is not None:
            # XXX add to features
            raise NotImplementedError()

        if filter_strides is not None:
            # XXX add to features
            raise NotImplementedError()

        if n_imgs != 1:
            # XXX add to features
            raise NotImplementedError()
        if border_mode != 'valid':
            # XXX add to features
            raise NotImplementedError()

    def __repr__(self):
        assigns = ['%s=%s' % (n, v) for v, n in self.feature_pairs()]
        return "ProblemSpec(%s)" % ", ".join(assigns)

    def __eq__(self, other):
        return (type(self) == type(other)
                and (self.feature_pairs() == other.feature_pairs()))

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((type(self),) + tuple(self.feature_pairs()))

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
            #'n_imgs',
            'height',
            'width',
            'depth',
            'n_filters',
            'filter_height',
            'filter_width',
            #'img_strides',
            #'filter_strides',
            #'border_mode',
            ]
        return [(getattr(self, name), name) for name in names]

    def feature_names(self):
        return zip(*self.feature_pairs())[1]

    def feature_values(self):
        return map(float, zip(*self.feature_pairs())[0])


    def is_valid(self):
        return self.out_height > 0 and self.out_width > 0

    def gflops(self):
        return (self.n_imgs
                * self.out_height * self.out_width * self.n_filters
                * self.filter_height * self.filter_width * self.depth
                * 2  # mul and add
                / (1000.**3.)) #return as giga float ops

    def image_shape(self):
        return (self.height, self.width, self.depth)

    def filters_shape(self):
        return (self.n_filters, self.filter_height, self.filter_width,
                self.depth)

    def output_shape(self):
        return (self.out_height, self.out_width, self.n_filters)


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

    def record(self, prob_spec, op_spec, speed, ref_speed):
        assert ref_speed is not None
        for ii, (pspec, ospec, s, rs) in enumerate(self._observations):
            if (pspec == prob_spec and ospec == op_spec):
                self._observations[ii][2] = .75 * s + .25 * speed
                self._observations[ii][3] = .75 * rs + .25 * ref_speed
        self._observations.append([prob_spec, op_spec, speed, ref_speed])

    def build_dtree_rec(self, features, targets, global_idxs, feature_names,
            min_improvement,
            min_split_size,
            min_n_valid,
            min_frac_valid,
            rng):

        assert len(features) == len(targets) == len(global_idxs)
        assert features.shape[1] == len(feature_names)
        n_valid = max(min_n_valid, int(min_frac_valid * len(features)))

        def make_this_a_leaf():
            return dict(
                    kind='leaf',
                    mean=np.mean(targets),
                    var=np.var(targets),
                    idxs=global_idxs)

        def sse(x):
            return np.var(x) * x.size

        if len(features) < 2 * min_split_size + n_valid:
            # -- we don't have enough data
            return make_this_a_leaf()

        all_idxs = np.arange(len(features))
        rng.shuffle(all_idxs)

        valid_idxs = all_idxs[:n_valid]
        train_idxs = all_idxs[n_valid:]

        orig_sse_valid = np.sum(
                (targets[valid_idxs] - np.mean(targets[train_idxs])) ** 2)
        if orig_sse_valid < min_improvement:
            return make_this_a_leaf()

        # -- find best split point over all features
        best_sse = float('inf')
        for i in xrange(features.shape[1]):
            train_features_i = features[train_idxs, i]
            order_i = np.argsort(train_features_i)

            sorted_target_i = targets[train_idxs][order_i]

            # XXX : do this in linear time instead of quadratic!
            for j in xrange(min_split_size,
                    len(train_features_i) - min_split_size):
                assert train_features_i[order_i[j - 1]] <= train_features_i[order_i[j]]
                if train_features_i[order_i[j - 1]] != train_features_i[order_i[j]]:
                    below = sorted_target_i[:j]
                    above = sorted_target_i[j:]
                    new_total_sse = sse(below) + sse(above)
                    if new_total_sse < best_sse:
                        split_pt = (0.5 * train_features_i[order_i[j - 1]]
                                + 0.5 * train_features_i[order_i[j]])
                        logger.debug('new best sse %f  (%i, %i) (%s < %s)' % (
                            new_total_sse,
                            i, j,
                            feature_names[i], split_pt))
                        best_ij = (i, j, split_pt,
                                np.mean(below), np.mean(above))
                        best_sse = new_total_sse

        if best_sse == float('inf'):
            # this can happen if the features are discrete, and such that it
            # isn't possible to leave min_split_size examples in a subtree,
            # even if there are more than 2 * min_split_size examples here.
            return make_this_a_leaf()

        ii, _jj, split_pt, mean_below, mean_above = best_ij
        # -- determine validation set performance
        new_sse_valid = 0
        for vi in valid_idxs:
            tvi = targets[vi]
            fvi = features[vi, ii]
            if fvi < split_pt:
                new_sse_valid += (tvi - mean_below) ** 2
            else:
                new_sse_valid += (tvi - mean_above) ** 2


        if new_sse_valid < (orig_sse_valid - min_improvement):
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
                        feature_names=feature_names,
                        min_improvement=min_improvement,
                        min_split_size=min_split_size,
                        min_n_valid=min_n_valid,
                        min_frac_valid=min_frac_valid,
                        rng=rng,
                        ),
                    right=self.build_dtree_rec(
                        features[rightidxs],
                        targets[rightidxs],
                        global_idxs[rightidxs],
                        feature_names=feature_names,
                        min_improvement=min_improvement,
                        min_split_size=min_split_size,
                        min_n_valid=min_n_valid,
                        min_frac_valid=min_frac_valid,
                        rng=rng,
                        ),
                    )
        else:
            return make_this_a_leaf()

    def build_dtree(self, rng, force=True):
        if not force:
            if (len(self._observations) < 7 or
                    len(self._observations) < 1.1 * self._dtree_n_obs):
                return
        if len(self._observations) == 0:
            return
        features = []
        targets = []
        logger.debug('building dtree from %i observations' %
                len(self._observations))
        for prob_spec, op_spec, speed, ref_speed in self._observations:
            feature = np.concatenate([
                prob_spec.feature_values(),
                op_spec.feature_values()])
            target = np.log(speed + 1e-10) - np.log(ref_speed + 1e-10)
            features.append(feature)
            targets.append(target)

        # just use last prob_spec
        feature_names = prob_spec.feature_names() + op_spec.feature_names()
        features = np.asarray(features, order='F')
        targets = np.asarray(targets)

        self._dtree = self.build_dtree_rec(features, targets,
                global_idxs=np.arange(len(features)),
                feature_names=feature_names,
                min_improvement=0,
                min_split_size=3,
                min_n_valid=5,
                min_frac_valid=.2,
                rng=rng,
                )
        self._dtree_n_obs = len(features)

        if 0:
            for i, o in enumerate(self._observations):
                print 'OBS', i, o[0]
                print 'OBS', i, o[1]
                print 'OBS', i, o[2]

    def print_dtree(self, node=None, prefix=""):
        if self._dtree is None:
            print 'DTREE undefined'
            return
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

    def _find_leaf(self, feature, node, prefix=""):
        if node['kind'] == 'fork':
            if feature[node['feature']] < node['value']:
                #print '-- branch ', node['feature_name'], '<', node['value']
                child = node['left']
            else:
                #print '-- branch ', node['feature_name'], '>=', node['value']
                child = node['right']
            return self._find_leaf(feature, child, prefix+"  ")
        else:
            return node

    def predict(self, prob_spec, op_spec):
        if self._dtree is None:
            return 0
        else:
            leaf = self._find_leaf(
                    feature=np.asarray(
                        prob_spec.feature_values() + op_spec.feature_values()),
                    node=self._dtree,
                    )
            return leaf['mean']


class Timing(object):
    def __init__(self, prob_spec, op_spec):
        self.prob_spec = prob_spec
        self.op_spec = op_spec
        self.valid = True
        self.timings = dict([(key, [])
                        for key in ('upload',
                                    #'set_up',
                                    'process',
                                    'cuda',
                                    'download',
                                    )])

    def stats(self):
        return dict(
                [(key, {
                    'median': scipy.median(t),
                    'mean': scipy.mean(t),
                    'std': scipy.std(t),
                    'max': max(t),
                    'min': min(t),
                    }) for key, t in self.timings.iteritems()])

    def speed(self, key='median'):
        stats = self.stats()
        gflop = self.prob_spec.gflops()
        gflops_cuda = {
            'median': gflop / stats['cuda']['median'],
            #'mean': gflop / stats['cuda']['mean'],
            ## mean is different if
            ## computed before or after division
            'max': gflop / stats['cuda']['min'],  # -- N.B. flip min -> max
            'min': gflop / stats['cuda']['max'],
            }
        return gflops_cuda[key]

    def measure_1(self, fop, in_, out_, fb_, in_data, fb_data, dotransfer=True):
        # -- upload data
        sys.stdout.flush()
        try:
            start = time.time()
            if dotransfer:
                in_[:] = in_data
                # XXX: writing 0 here is important for correctness!
                out_[:] = 0
                fb_[:] = fb_data
            end = time.time()
            t_upload = end - start

            # -- process convolution
            # Note Bene: Filter != Conv
            start = time.time()
            try:
                t_cuda = fop()
            except fbconv3_cuda.InvalidConfig, e:
                logger.debug('-- InvalidConfig %s' % str(e))
                self.valid = False
                return
            end = time.time()
            t_process = end - start

            start = time.time()
            if dotransfer:
                out_data = out_[:]
            else:
                out_data = None
            end = time.time()
            t_download = end - start

            self.timings['upload'] += [t_upload]
            self.timings['process'] += [t_process]
            self.timings['cuda'] += [t_cuda]
            self.timings['download'] += [t_download]
        except pycuda._driver.LaunchError:
            # memcpy can fail, call can fail
            self.valid = False

    def get_sample_data(self):

        img_shp = self.prob_spec.image_shape()
        ker_shp = self.prob_spec.filters_shape()
        out_shp = self.prob_spec.output_shape()

        data_rng = np.random.RandomState(12345)
        # XXX: put random numbers or arange mod something to check correctness
        in_data = np.zeros(img_shp, dtype='float32')
        fb_data = np.zeros(ker_shp, dtype='float32')
        out_data = np.empty(out_shp, dtype='float32')
        return in_data, fb_data, out_data

    def measure_setup(self, context):
        img_shp = self.prob_spec.image_shape()
        ker_shp = self.prob_spec.filters_shape()
        out_shp = self.prob_spec.output_shape()

        in_ = fbconv3_cuda.Input(*img_shp)
        fb_ = fbconv3_cuda.Filterbank(*ker_shp)
        out_ = fbconv3_cuda.Output(*out_shp)

        # -- set-up operation (i.e. compilation)
        fb_[:] = 0
        fop = self.op_spec.FilterOp(in_, fb_, out_, ctxt=context)
        return fop, in_, fb_, out_

    def measure(self, device, N=8):
        """
        Add N timing measurements to self.timings
        """
        in_data, fb_data, out_dat = self.get_sample_data()

        with CudaContext(device) as context:
            try:
                fop, in_, fb_, out_ = self.measure_setup(context)
            except (fbconv3_cuda.InvalidConfig,
                    pycuda._driver.LogicError,    #XXX: cuModuleGetTexRef not found
                    pycuda.driver.CompileError,), e: #XXX: using too much shared memory
                logger.debug('-- InvalidConfig %s' % str(e))
                self.valid = False
                return
            for i in xrange(N):
                self.measure_1(fop, in_, out_, fb_, in_data, fb_data)


def quick_winner(new_timing, timing, device, finding):
    in_data, fb_data, out_dat = new_timing.get_sample_data()
    with CudaContext(device) as context:
        try:
            fop, in_, fb_, out_ = new_timing.measure_setup(context)
        except (fbconv3_cuda.InvalidConfig,
                pycuda._driver.LogicError,    #XXX: cuModuleGetTexRef not found
                pycuda.driver.CompileError,), e: #XXX: using too much shared memory
            logger.debug('-- InvalidConfig %s' % str(e))
            new_timing.valid = False
        i = 0
        while i < 10 and new_timing.valid:
            if i >= 3 and new_timing.speed('max') < .5 * timing.speed():
                break
            new_timing.measure_1(fop, in_, out_, fb_, in_data, fb_data,
                    dotransfer=(i == 0))
            i += 1
    finding['qw_%i' % len(finding)] = new_timing
    if new_timing.valid and new_timing.speed() > timing.speed():
        return new_timing
    else:
        return timing


def gcg_grid_autotune(timing, device, finding):
    import fbconv3_cuda_metaparams_cherrypick
    metaparams_list = fbconv3_cuda_metaparams_cherrypick.metaparams_list

    results = []
    for mp in metaparams_list:
        op_spec = OpSpec(use_fast_math=False, **mp)
        new_timing = Timing(timing.prob_spec, op_spec)
        timing = quick_winner(new_timing, timing, device, finding)
    return timing


def genetic_step(timing, device, mutation_rate, rng, finding):
    assert timing.valid
    candidate = resample_some_coords(timing.op_spec, rng, mutation_rate)
    new_timing = Timing(timing.prob_spec, candidate)
    return quick_winner(new_timing, timing, device, finding)


def tree_step(timing, device, rng, wisdom, ref_speed, N, mutation_rate):
    #print 'building tree'
    t0 = time.time()
    wisdom.build_dtree(rng)
    print 'building dtree took', time.time() - t0
    #wisdom.print_dtree()
    #print 'searching for candidate'
    if wisdom._dtree is None:
        candidate = resample_some_coords(timing.op_spec, rng, mutation_rate)
    else:
        candidate = timing.op_spec
        candidate_score = wisdom.predict(
                timing.prob_spec,
                candidate)
        for i in range(N):
            candidate2 = resample_some_coords(candidate, rng, mutation_rate)
            candidate2_score = wisdom.predict(
                    timing.prob_spec,
                    candidate2)
            if candidate2_score >= candidate_score:
                # print '--  cand score', candidate2_score
                candidate = candidate2
                candidate_score = candidate2_score
    new_timing = Timing(timing.prob_spec, candidate)
    #print 'timing (candidate score)', candidate_score
    new_timing.measure(device)
    if new_timing.valid:
        wisdom.record(new_timing.prob_spec, new_timing.op_spec,
                new_timing.speed(), ref_speed)
    else:
        wisdom.record(new_timing.prob_spec, new_timing.op_spec,
                0.8 * ref_speed, ref_speed)

    if new_timing.valid and new_timing.speed() > timing.speed():
        return new_timing
    else:
        return timing



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

    encumbent_speed = ref_speed = self.measure_speed(encumbent,
            n_warmups=2, n_runs=8, ref_speed=None, wisdom=wisdom,
            device=device)

    n_clocked = [0]
    def clock_candidate():
        n_clocked[0] += 1
        return self.measure_speed(candidate,
                n_warmups=2,
                n_runs=8,
                ref_speed=ref_speed,
                abort_rel_thresh=0.5,
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
        candidate = resample_some_coords(encumbent, rng, .25)
        candidate_speed = clock_candidate()
        if candidate_speed > 0:
            if candidate_speed > encumbent_speed:
                encumbent = candidate
                encumbent_speed = candidate_speed
    #print 'N_CLOCKED', n_clocked[0]
    return encumbent


def assert_correct(prob_spec, op):
    """raise assertion error if op() produces incorrect output
    """
    raise NotImplementedError()


