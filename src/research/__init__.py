"""Research library: model and signal functions shared by the research notebook and the strategy classes.

Timing convention (same as the engine): a signal indexed at date T is what a strategy can compute at
3:45PM on T, i.e. from EOD data through T-1 and the day-T 3PM snapshot. It is scored against returns
from the close of T onward.
"""
