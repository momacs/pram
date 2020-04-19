from pram.entity import Group
from pram.rule   import FibonacciSeq
from pram.sim    import Simulation


(Simulation().
    add([
        FibonacciSeq('fib'),
        Group(m=1, attr={ 'fib': 0 })
    ]).
    run(20).
    summary(False, 128,0,0,0)
)
