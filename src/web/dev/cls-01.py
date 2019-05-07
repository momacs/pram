'''
Experiment with getting info from classes and instances passed as argument.

References
    https://docs.python.org/2/library/inspect.html#inspect.cleandoc
    https://docs.python.org/3/library/textwrap.html
    https://www.python.org/dev/peps/pep-0257    (scroll down for code)
'''

import inspect


# Preceding comment
# Second line
class A(object):  # note the line number
    '''
    Class description goes here.

    Multiline comments (like this one) are ubiquitous and should be handled gracefully.  Multiple lines are especially
    common in descriptions of possibly complex mechanisms implemented by the PRAM rules.  It is therefore imperative
    that such cases receive proper attention.
    '''

    def __init__(self, name=None):
        ''' Not a docstring! '''

        self.name = name  # also not a docstring!
        pass              # this is the 14th line of code


def get_inf(x):
    cls = x if inspect.isclass(x) else x.__class__
    print(cls.__name__)
    print('---')

    print(inspect.cleandoc(cls.__doc__).split('\n'))
    print('---')

    print(inspect.getcomments(cls))
    print('---')

    src = inspect.getsourcelines(cls)
    print(f'File line: {src[1]}')
    print(f'Line count (including comments): {len(src[0])}')
    print('Code:')
    print(''.join(src[0]))


get_inf(A)

# get_inf(A())
# get_inf(A('test'))
