# http://www.pgbovine.net/cpython-internals.htm
# https://www.youtube.com/watch?v=HVUTjQzESeo

# https://www.programcreek.com/python/

# https://docs.python.org/3.6/library/ast.html
# https://docs.python.org/3.6/library/dis.html
# https://docs.python.org/3.6/library/parser.html

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from pram.entity import AttrFluStage, GroupSplitSpec
from pram.rule   import GotoRule, Rule, TimeInt


# ----------------------------------------------------------------------------------------------------------------------
class ProgressFluRule(Rule):
    def __init__(self, t=TimeInt(8,20), memo=None):
        super().__init__('progress-flu', t, memo)

    def apply(self, pop, group, iter, t):
        p_infection = 0.05

        if group.get_attr('flu-stage') == AttrFluStage.NO:
        # if group.has_attr({ 'flu-stage': AttrFluStage.NO }):
            return [
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu-stage': AttrFluStage.NO     }),
                GroupSplitSpec(p=p_infection,     attr_set={ 'flu-stage': AttrFluStage.ASYMPT }),
                GroupSplitSpec(p=0.00,            attr_set={ 'flu-stage': AttrFluStage.SYMPT  })
            ]
        elif group.get_attr('flu-stage') == AttrFluStage.ASYMPT:
        # elif group.has_attr({ 'flu-stage': AttrFluStage.ASYMPT }):
            return [
                GroupSplitSpec(p=0.20, attr_set={ 'flu-stage': AttrFluStage.NO     }),
                GroupSplitSpec(p=0.00, attr_set={ 'flu-stage': AttrFluStage.ASYMPT }),
                GroupSplitSpec(p=0.80, attr_set={ 'flu-stage': AttrFluStage.SYMPT  })
            ]
        elif group.get_attr('flu-stage') == AttrFluStage.SYMPT:
        # elif group.has_attr({ 'flu-stage': AttrFluStage.SYMPT }):
            return [
                GroupSplitSpec(p=0.05, attr_set={ 'flu-stage': AttrFluStage.NO     }),
                GroupSplitSpec(p=0.20, attr_set={ 'flu-stage': AttrFluStage.ASYMPT }),
                GroupSplitSpec(p=0.75, attr_set={ 'flu-stage': AttrFluStage.SYMPT  })
            ]
        else:
            raise ValueError("Invalid value for attribute 'flu-stage'.")

    def is_applicable(self, group, iter, t):
        attr = 'flu-stage'
        # return super().is_applicable(iter, t) and group.has_attr([ 'flu-stage' ])
        return super().is_applicable(iter, t) and group.has_attr([ attr ])


# ----------------------------------------------------------------------------------------------------------------------
class R(Rule):
    def __init__(self, t=TimeInt(8,20), memo=None):
        super().__init__('progress-flu', t, memo)

    def an(self, s): return f'b{s}'  # attribute name
    def rn(self, s): return f's{s}'  # relation  name

    def apply(self, group, iter, t):
        if group.has_attr({ 'flu-stage': AttrFluStage.NO }):
            pass
        elif group.has_attr({ 'flu-stage': AttrFluStage.ASYMPT }):
            pass
        elif group.has_attr({ 'flu-stage': AttrFluStage.SYMPT }):
            pass

    def is_applicable(self, group, iter, t):
        g = group
        c01, c02, c03, c04, c05 = 'cc01', 'cc02', 'cc03', 'cc04', 'cc05'  # attribute names stored in local variables
        s01, s02, s03, s04, s05 = 'ss01', 'ss02', 'ss03', 'ss04', 'ss05'  # ^ (relation)

        return (
            super().is_applicable(iter, t) and

            g.has_attr('a01') and g.has_attr([ 'a02', 'a03' ]) and g.has_attr({ 'a04':1, 'a05':2 }) and
            g.has_attr(c01) and g.has_attr([ c02, c03 ]) and g.has_attr({ c04:1, c05:2 }) and
            g.has_attr(self.an('01')) and g.has_attr([ self.an('02'), self.an('03') ]) and g.has_attr({ self.an('04'):1, self.an('05'):2 }) and

            g.has_rel('r01') and g.has_rel([ 'r02', 'r03' ]) and g.has_rel({ 'r04':1, 'r05':2 }) and
            g.has_rel(s01) and g.has_rel([ s02, s03 ]) and g.has_rel({ s04:1, s05:2 }) and
            g.has_rel(self.rn('01')) and g.has_rel([ self.rn('02'), self.rn('03') ]) and g.has_rel({ self.rn('04'):1, self.rn('05'):2 })
        )


# ----------------------------------------------------------------------------------------------------------------------
# (1) Syntax trees:

import ast
import inspect

from collections import Counter


class RuleAnalyzer(object):
    '''
    Analyzes the syntax (i.e., abstract syntax trees or ASTs) of rule objects to identify group attributes and
    relations these rules condition on.

    References
        https://docs.python.org/3.6/library/dis.html
        https://docs.python.org/3.6/library/inspect.html
        https://docs.python.org/3.6/library/ast.html

        https://github.com/hchasestevens/astpath
        https://astsearch.readthedocs.io/en/latest
    '''

    def __init__(self):
        self.attr = set()
        self.rel  = set()

        self.cnt_known   = Counter({ 'get_attr': 0, 'get_rel': 0, 'has_attr': 0, 'has_rel': 0 })
        self.cnt_unknown = Counter({ 'get_attr': 0, 'get_rel': 0, 'has_attr': 0, 'has_rel': 0 })

    def _dump(self, node, annotate_fields=True, include_attributes=False, indent='  '):
        '''
        Source: https://bitbucket.org/takluyver/greentreesnakes/src/default/astpp.py?fileviewer=file-view-default
        '''

        if not isinstance(node, ast.AST):
            raise TypeError('expected AST, got %r' % node.__class__.__name__)
        return self._format(node, 0, annotate_fields, include_attributes, indent)

    def _format(self, node, level=0, annotate_fields=True, include_attributes=False, indent='  '):
        '''
        Source: https://bitbucket.org/takluyver/greentreesnakes/src/default/astpp.py?fileviewer=file-view-default
        '''

        if isinstance(node, ast.AST):
            fields = [(a, self._format(b, level, annotate_fields, include_attributes, indent)) for a, b in ast.iter_fields(node)]
            if include_attributes and node._attributes:
                fields.extend([(a, self._format(getattr(node, a), level, annotate_fields, include_attributes, indent))
                               for a in node._attributes])
            return ''.join([
                node.__class__.__name__,
                '(',
                ', '.join(('%s=%s' % field for field in fields)
                           if annotate_fields else
                           (b for a, b in fields)),
                ')'])
        elif isinstance(node, list):
            lines = ['[']
            lines.extend((indent * (level + 2) + self._format(x, level + 2, annotate_fields, include_attributes, indent) + ','
                         for x in node))
            if len(lines) > 1:
                lines.append(indent * (level + 1) + ']')
            else:
                lines[-1] += ']'
            return '\n'.join(lines)
        return repr(node)

    def _analyze_test(self, node):
        if isinstance(node, ast.AST):
            fields = [(a, self._analyze_test(b)) for a,b in ast.iter_fields(node)]
            return ''.join([node.__class__.__name__, '(', ', '.join((b for a,b in fields)), ')'])
        elif isinstance(node, list):
            lines = []
            lines.extend((self._analyze_test(x) + ',' for x in node))
            return '\n'.join(lines)
        return repr(node)

    def _analyze(self, node):
        '''
        Processe a node of the AST recursively looking for method calls that suggest group attribute and relation names
        conditioned on by the rule.  It also updates counts of all known and unknown names (compartmentalized by the
        method name).

        References
            https://docs.python.org/3.6/library/ast.html#abstract-grammar
        '''

        if isinstance(node, ast.AST):
            for _,v in ast.iter_fields(node):
                self._analyze(v)

            if node.__class__.__name__ == 'Call':
                call_args = list(ast.iter_fields(node))[1][1]

                if list(ast.iter_fields(node))[0][1].__class__.__name__ == 'Attribute':
                    attr = list(ast.iter_fields(node))[0][1]
                    attr_name = list(ast.iter_fields(attr))[1][1]

                    if attr_name in ('get_attr', 'get_rel', 'has_attr', 'has_rel'):
                        call_args = call_args[0]
                        if call_args.__class__.__name__ == 'Str':
                            if attr_name in ('get_attr', 'has_attr'):
                                self.attr.add(RuleAnalyzer.get_str(call_args))
                            else:
                                self.rel.add(RuleAnalyzer.get_str(call_args))
                            self.cnt_known[attr_name] += 1
                        elif call_args.__class__.__name__ in ('List', 'Dict'):
                            for i in list(ast.iter_fields(call_args))[0][1]:
                                if i.__class__.__name__ == 'Str':
                                    if attr_name in ('get_attr', 'has_attr'):
                                        self.attr.add(RuleAnalyzer.get_str(i))
                                    else:
                                        self.rel.add(RuleAnalyzer.get_str(i))
                                    self.cnt_known[attr_name] += 1
                                else:
                                    self.cnt_unknown[attr_name] += 1
                                    # print(list(ast.iter_fields(i)))
        elif isinstance(node, list):
            for i in node:
                self._analyze(i)

    def analyze(self, rule):
        tree = ast.fix_missing_locations(ast.parse(inspect.getsource(rule.__class__)))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef): continue  # skip non-classes

            for node_fn in node.body:
                if not isinstance(node_fn, ast.FunctionDef): continue  # skip non-methods
                self._analyze(node_fn.body)

                # if node_fn.name in ('is_applicable'): print(self._analyze_01(node_fn.body))

    def dump(self, rule):
        tree = ast.fix_missing_locations(ast.parse(inspect.getsource(rule.__class__)))
        print(self._dump(tree))

    @staticmethod
    def get_str(node):
        return list(ast.iter_fields(node))[0][1]


from pram.rule import GotoRule, TimeInt
from rules import AttendSchoolRule, ProgressFluRule, ProgressAndTransmitFluRule

ra = RuleAnalyzer()

# ra.analyze(ProgressFluRule())
# ra.analyze(R())

ra.analyze(GotoRule(TimeInt( 8,12), 0.4, 'home',  'work'))

# ra.analyze(ProcessRule(AttendSchoolRule())
# ra.analyze(ProcessRule(ProgressFluRule())
# ra.analyze(ProcessRule(ProgressAndTransmitFluRule())

# ra.dump(ProgressFluRule())
# ra.dump(R())

print('----')
print(f'attr    : {ra.attr}')
print(f'rel     : {ra.rel}')
print(f'known   : {ra.cnt_known}')
print(f'unknown : {ra.cnt_unknown}')


# ----
# import inspect
# print(inspect.getclasstree([ProgressFluRule]))


# ----
# import re
#
# CLS_RULE_PATT = re.compile('^class\s+([a-zA-Z0-9_]+)\s*\(Rule\)\s*:')
#
# with open(__file__) as f:
#     ln = f.readline()
#     if CLS_RULE_PATT.match(ln):
#         print(ln)


# ----
# print(R.__dict__['is_applicable'])

# import ast
# import inspect
#
# # code = ast.parse(inspect.getsource(R.__dict__['is_applicable']))
# code = ast.parse(inspect.getsource(R))
# print(ast.dump(code))

# parseprint('-a')


# ----
# https://julien.danjou.info/python-ast-checking-method-declaration/


# ----------------------------------------------------------------------------------------------------------------------
# (2) Disassembly:

# import dis
#
# def list_func_calls(fn):
#     ''' Source: https://stackoverflow.com/a/51904019 '''
#
#     funcs = []
#     bytecode = dis.Bytecode(fn)
#     instrs = list(reversed([instr for instr in bytecode]))
#     for (i, instr) in enumerate(instrs):
#         if instr.opname == 'CALL_FUNCTION':
#             # load_func_instr = instrs[i + instr.arg + 1]
#             # funcs.append(load_func_instr.argval)
#
#             method_name = instrs[i + instr.arg + 1].argval
#             method_args = ''
#             funcs.append(f'{method_name}({method_args})')
#
#     return ['%s' % funcname for funcname in reversed(funcs)]
#
# # print(list_func_calls(ProgressFluRule.apply))
# # print(list_func_calls(ProgressFluRule.is_applicable))
#
# print(list_func_calls(R.is_applicable))
#
# sys.exit(99)


# ----
# import dis
# dis.dis(ProgressFluRule)
# dis.dis(ProgressFluRule.apply)
# dis.dis(ProgressFluRule.is_applicable)
#
# sys.exit(99)


# ----
# import dis
# import sys
# from contextlib import contextmanager
#
# class AttrDict(dict):
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self
#
# @contextmanager # https://stackoverflow.com/a/12111817/2422125
# def captureStdOut(output):
#     stdout = sys.stdout
#     sys.stdout = output
#     try:
#         yield
#     finally:
#         sys.stdout = stdout
#
# """ for Python <3.4 """
# def get_instructions(func):
#     import StringIO
#
#     out = StringIO.StringIO()
#     with captureStdOut(out):
#         dis.dis(func)
#
#     return [AttrDict({
#                'opname': i[16:36].strip(),
#                'arg': int(i[37:42].strip() or 0),
#                'argval': i[44:-1].strip()
#            }) for i in out.getvalue().split("\n")]
#
# if sys.version_info < (3, 4):
#     dis.get_instructions = get_instructions
#     import __builtin__ as builtin
# else:
#     import builtins as builtin
#
#
# def get_function_calls(func, built_ins=False):
#     # the used instructions
#     ins = list(dis.get_instructions(func))[::-1]
#
#     # dict for function names (so they are unique)
#     names = {}
#
#     # go through call stack
#     for i, inst in list(enumerate(ins))[::-1]:
#         # find last CALL_FUNCTION
#         if inst.opname[:13] == "CALL_FUNCTION":
#
#             # function takes ins[i].arg number of arguments
#             ep = i + inst.arg + (2 if inst.opname[13:16] == "_KW" else 1)
#
#             # parse argument list (Python2)
#             if inst.arg == 257:
#                 k = i+1
#                 while k < len(ins) and ins[k].opname != "BUILD_LIST":
#                     k += 1
#
#                 ep = k-1
#
#             # LOAD that loaded this function
#             entry = ins[ep]
#
#             # ignore list comprehensions / ...
#             name = str(entry.argval)
#             if "." not in name and entry.opname == "LOAD_GLOBAL" and (built_ins or not hasattr(builtin, name)):
#                 # save name of this function
#                 names[name] = True
#
#             # reduce this CALL_FUNCTION and all its paramters to one entry
#             ins = ins[:i] + [entry] + ins[ep + 1:]
#
#     return sorted(list(names.keys()))
#
#
# print(get_function_calls(ProgressFluRule.is_applicable))
# print(get_function_calls(ProgressFluRule.apply))


# ----
# from inspect import getargvalues

# import inspect
# args, _, _, values = inspect.getargvalues(inspect.currentframe())
# print(values)

# r = ProgressFluRule()
# r.apply(None, Group(), 0, 0)


# ----
# https://github.com/rocky/python-uncompyle6
