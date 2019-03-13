# http://www.pgbovine.net/cpython-internals.htm
# https://www.youtube.com/watch?v=HVUTjQzESeo

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

    def site_x(self):
        return 'site-x'

    def apply(self, group, iter, t):
        pass

    def is_applicable(self, group, iter, t):
        g = group
        a = 'flu-stage'
        return super().is_applicable(iter, t) and g.has_attr([ a ]) and g.has_rel({ 'site': self.site_x() })


# ----------------------------------------------------------------------------------------------------------------------
# (1) Disassembly:

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


# ----------------------------------------------------------------------------------------------------------------------
# (2) Syntax trees:

import ast
import inspect


class ProcessRule(object):
    def __init__(self, rule):
        # self.tree = ast.fix_missing_locations(ast.parse(inspect.getsource(rule.__class__)))
        self.tree = ast.parse(inspect.getsource(rule.__class__))

    def _format(self, node, level=0, annotate_fields=True, include_attributes=False, indent='  '):
        ''' Source: https://bitbucket.org/takluyver/greentreesnakes/src/default/astpp.py?fileviewer=file-view-default '''

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

    def _dump(self, node, annotate_fields=True, include_attributes=False, indent='  '):
        ''' Source: https://bitbucket.org/takluyver/greentreesnakes/src/default/astpp.py?fileviewer=file-view-default '''

        if not isinstance(node, ast.AST):
            raise TypeError('expected AST, got %r' % node.__class__.__name__)
        return self._format(node, 0, annotate_fields, include_attributes, indent)

    def dump(self):
        print(self._dump(self.tree))

    def _proc_method_node(self, node):
        if isinstance(node, ast.AST):
            fields = [(a, self._proc_method_node(b)) for a,b in ast.iter_fields(node)]
            return ''.join([node.__class__.__name__, '(', ', '.join((b for a,b in fields)), ')'])
        elif isinstance(node, list):
            lines = []
            lines.extend((self._proc_method_node(x) + ',' for x in node))
            return '\n'.join(lines)
        return repr(node)

    def proc(self):
        for stmt in ast.walk(self.tree):
            if not isinstance(stmt, ast.ClassDef): continue  # skip non-classes

            for i in stmt.body:
                if not isinstance(i, ast.FunctionDef): continue  # skip non-methods
                if i.name in ('is_applicable'):
                    print(self._proc_method_node(i.body))


# pr = ProcessRule(ProgressFluRule())
pr = ProcessRule(R())
# pr.dump()
pr.proc()


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
# print(ProgressFluRule.__dict__['is_applicable'])
#
# import ast
# code = ast.parse(ProgressFluRule.__dict__['is_applicable'])
# print(ast.dump(code))

# parseprint('-a')


# ----
# https://julien.danjou.info/python-ast-checking-method-declaration/
