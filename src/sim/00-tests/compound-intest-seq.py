from pram.entity import Group
from pram.rule   import CompoundInterstSeq
from pram.sim    import Simulation


(Simulation().
    add([
        CompoundInterstSeq('x', 0.043, 4),
        Group(m=1, attr={ 'x': 1500 })
    ]).
    run(6).
    summary(False, 128,0,0,0)  # after 6 years of 4.3% interest, x should be 1938.84
)
