"""
CFR algorithms module
"""
from .vanilla import VanillaCFR
from .cfr_plus import CFRPlus
from .dcfr import DiscountedCFR

__all__ = ['VanillaCFR', 'CFRPlus', 'DiscountedCFR']