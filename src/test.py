# https://docs.python.org/3.6/library/unittest.html
#
# python -m unittest discover

import unittest

from pram import AttrFluStatus, AttrSex, Group, Site


class GroupTestCase(unittest.TestCase):
    def test_attributes_and_relations(self):
        f = self.assertFalse
        t = self.assertTrue

        g0 = Group('g.0', 100)
        g1 = Group('g.1', 200, { 'sex': 'f', 'income': 'l' }, { 'location': 'home' })

        f(g0.has_attr([ 'sex' ]))
        f(g0.has_attr({ 'sex': 'f' }))
        f(g0.has_attr({ 'sex': 'f', 'income': 'l' }))

        f(g0.has_rel({ 'location': 'home' }))

        t(g1.has_attr([ 'sex' ]))
        t(g1.has_attr({ 'sex': 'f' }))
        t(g1.has_attr({ 'sex': 'f', 'income': 'l' }))
        f(g1.has_attr({ 'sex': 'f', 'income': 'h' }))

        t(g1.has_rel({ 'location': 'home' }))
        f(g1.has_rel({ 'location': 'work' }))

    def test_comparisons(self):
        eq = self.assertEqual
        ne = self.assertNotEqual

        eq(Group(),            Group())             # different objects
        eq(Group('g.10', 100), Group('g.20', 100))  # different names
        eq(Group('g.10',  11), Group('g.20',  22))  # different sizes

        eq(Group(attr={}), Group())           # no attributes
        eq(Group(attr={}), Group(attr={}))    # no attributes
        eq(Group(attr={}), Group(attr=None))  # no attributes

        eq(Group(attr={ 'sex': 'f' }), Group(attr={ 'sex': 'f' }))  # same attributes (primitive data types)
        eq(Group(attr={ 'age': 99 }),  Group(attr={ 'age': 99 }))   # same attributes (primitive data types)

        eq(Group(attr={ 'sex': AttrSex.f }),        Group(attr={ 'sex': AttrSex.f }))            # same attributes (composite data types)
        eq(Group(attr={ 'sex': AttrFluStatus.no }), Group(attr={ 'sex': AttrFluStatus.no }))     # same attributes (composite data types)
        ne(Group(attr={ 'sex': AttrFluStatus.no }), Group(attr={ 'sex': AttrFluStatus.sympt }))  # different attributes (composite data types)

        ne(Group(attr={ 'sex': 'f' }), Group(attr={ 'xes': 'f' }))  # different attribute keys
        ne(Group(attr={ 'sex': 'f' }), Group(attr={ 'sex': 'm' }))  # different attribute values

        eq(Group(attr={ 'sex': 'f', 'income': 'l' }), Group(attr={ 'income': 'l', 'sex': 'f' }))  # same attributes, different order


class SiteTestCase(unittest.TestCase):
    def test_comparisons(self):
        eq = self.assertEqual
        ne = self.assertNotEqual

        eq(Site('a'), Site('a'))  # different objects, same name
        ne(Site('a'), Site('b'))  # different objects, different name


class SimulationTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_(self):
        pass


if __name__ == '__main__':
    unittest.main()  # verbosity=2
