"""
Test CaseIteratorDriver.
"""

import glob
import logging
import os.path
import pkg_resources
import shutil
import sys
import unittest

import numpy.random

from enthought.traits.api import Float, Array, TraitError

from openmdao.main.api import Assembly, Component, Case, ListCaseIterator, set_as_top
from openmdao.lib.drivers.caseiterdriver import CaseIteratorDriver
import openmdao.util.testutil

ORIG_DIR = os.getcwd()

# pylint: disable-msg=E1101


def rosen_suzuki(x):
    """ Evaluate polynomial from CONMIN manual. """
    return x[0]**2 - 5.*x[0] + x[1]**2 - 5.*x[1] + \
           2.*x[2]**2 - 21.*x[2] + x[3]**2 + 7.*x[3] + 50


class DrivenComponent(Component):
    """ Just something to be driven and compute results. """

    x = Array('d', value=[1., 1., 1., 1.], iostatus='in')
    y = Array('d', value=[1., 1., 1., 1.], iostatus='in')
    rosen_suzuki = Float(0., iostatus='out')
    sum_y = Float(0., iostatus='out')
        
    def __init__(self, *args, **kwargs):
        super(DrivenComponent, self).__init__(*args, **kwargs)

    def execute(self):
        """ Compute results from input vector. """
        self.rosen_suzuki = rosen_suzuki(self.x)
# This gets "iter() returned non-iterator of type 'SyncNetProxy'"
# (RPyC 2.6 doesn't know to 'box' Array objects)
#        self.sum_y = sum(self.y)
        self.sum_y = 0
        for i in range(len(self.y)):
            self.sum_y += self.y[i]


class MyModel(Assembly):
    """ Use CaseIteratorDriver with DrivenComponent. """

    #name='CID_TestModel', 
    def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)
        self.add_container('driver', CaseIteratorDriver())
        self.driver.add_container('model', DrivenComponent())


class DriverTestCase(unittest.TestCase):
    """ Test CaseIteratorDriver. """

    directory = pkg_resources.resource_filename('openmdao.lib.drivers', 'test')

    def setUp(self):
        os.chdir(self.directory)
        self.model = set_as_top(MyModel())
        self.cases = []
        for i in range(10):
            inputs = [('x', None, numpy.random.normal(size=4)),
                      ('y', None, numpy.random.normal(size=10))]
            outputs = [('rosen_suzuki', None, None),
                       ('sum_y', None, None)]
            self.cases.append(Case(inputs, outputs))

    def tearDown(self):
        self.model.pre_delete()
        self.model = None
        for server_dir in glob.glob('LocalHost_*'):
            shutil.rmtree(server_dir)
        end_dir = os.getcwd()
        os.chdir(ORIG_DIR)
        if end_dir != self.directory:
            self.fail('Ended in %s, expected %s' % (end_dir, self.directory))

    def test_sequential(self):
        logging.debug('')
        logging.debug('test_sequential')
        self.run_cases(sequential=True)

    def test_concurrent(self):
        logging.debug('')
        logging.debug('test_concurrent')
        try:
            self.run_cases(sequential=False, n_servers=5)
        except NotImplementedError, exc:
            msg = 'driver: Concurrent evaluation is not' \
                  ' supported yet.'
            self.assertEqual(str(exc), msg)
        else:
            self.fail('Expected NotImplementedError')

    def run_cases(self, sequential, n_servers=0):
        self.model.driver.sequential = sequential
        self.model.driver._n_servers = n_servers
        self.model.driver.iterator = ListCaseIterator(self.cases)
        results = []
        self.model.driver.outerator = results

        self.model.run()

        self.assertEqual(len(results), len(self.cases))
        for case in results:
            self.assertEqual(case.msg, None)
            self.assertEqual(case.outputs[0][2],
                             rosen_suzuki(case.inputs[0][2]))
            self.assertEqual(case.outputs[1][2],
                             sum(case.inputs[1][2]))

    def test_save_load(self):
        logging.debug('')
        logging.debug('test_save_load')

        self.model.driver.iterator = ListCaseIterator(self.cases)
        results = []
        self.model.driver.outerator = results

        # Set local dir in case we're running in a different directory.
        py_dir = self.directory
        python = openmdao.util.testutil.find_python('openmdao.lib')
        retcode = self.model.check_save_load(py_dir=py_dir, python=python)
        self.assertEqual(retcode, 0)

    def test_noinput(self):
        logging.debug('')
        logging.debug('test_noinput')

        cases = []
        for i in range(2):
            inputs = [('x', None, numpy.random.normal(size=4)),
                      ('z', None, numpy.random.normal(size=10))]
            outputs = [('rosen_suzuki', None, None),
                       ('sum_y', None, None)]
            cases.append(Case(inputs, outputs))

        self.model.driver.iterator = ListCaseIterator(cases)
        results = []
        self.model.driver.outerator = results

        self.model.run()

        self.assertEqual(len(results), len(cases))
        msg = "driver: Exception setting 'z':" \
              " driver.model: object has no attribute 'z'"
        for case in results:
            self.assertEqual(case.msg, msg)

    def test_nooutput(self):
        logging.debug('')
        logging.debug('test_nooutput')

        cases = []
        for i in range(2):
            inputs = [('x', None, numpy.random.normal(size=4)),
                      ('y', None, numpy.random.normal(size=10))]
            outputs = [('rosen_suzuki', None, None),
                       ('sum_z', None, None)]
            cases.append(Case(inputs, outputs))

        self.model.driver.iterator = ListCaseIterator(cases)
        results = []
        self.model.driver.outerator = results

        self.model.run()

        self.assertEqual(len(results), len(cases))
        msg = "driver: Exception getting 'sum_z':" \
              " driver.model: object has no attribute 'sum_z'"
        for case in results:
            self.assertEqual(case.msg, msg)

    def test_noiterator(self):
        logging.debug('')
        logging.debug('test_noiterator')

        self.model.driver.outerator = []
        try:
            self.model.run()
        except TraitError, exc:
            msg = "driver: required plugin 'iterator' is not" \
                  " present"
            self.assertEqual(str(exc), msg)
        else:
            self.fail('TraitError expected')

    def test_noouterator(self):
        logging.debug('')
        logging.debug('test_noouterator')

        self.model.driver.iterator = ListCaseIterator([])
        try:
            self.model.run()
        except TraitError, exc:
            msg = "driver: required plugin 'outerator' is not" \
                  " present"
            self.assertEqual(str(exc), msg)
        else:
            self.fail('TraitError expected')


if __name__ == "__main__":
    import nose
    sys.argv.append('--cover-package=openmdao')
    sys.argv.append('--cover-erase')
    nose.runmodule()

