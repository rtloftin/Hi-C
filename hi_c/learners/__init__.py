
def get_naive():
    from hi_c.learners.gradient import NaiveLearner
    return NaiveLearner


def get_lola():
    from hi_c.learners.lola import LOLA
    return LOLA


def get_hierarchical():
    from hi_c.learners.hierarchical import HierarchicalGradient
    return HierarchicalGradient


def get_hi_c():
    from hi_c.learners.hi_c import HiC
    return HiC


LEARNERS = {
    "naive": get_naive,
    "lola": get_lola,
    "hierarchical": get_hierarchical,
    "hi_c": get_hi_c,
}


def get_learner_class(name):
    if name not in LEARNERS:
        raise ValueError(f"Learner '{name}' is not defined")
    
    return LEARNERS[name]()


__all__ = [get_learner_class]
