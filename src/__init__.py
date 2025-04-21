"""
PRISM - Periodicity Recognition through Index & Sector Market-analysis
----------------------------------------------------------------------
User can do:

    >>> import prism as ps
    >>> ps.run_pipeline()          # main analysis
    >>> ps.fetch_all()             # data update

The work is done in fetch.py, preprocess.py, fourier.py, etc.
"""

from .main import run_pipeline
from .fetch import fetch_all
