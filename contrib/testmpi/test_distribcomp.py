
from unittest import TestCase
import time
import sys

import numpy as np

from openmdao.lib.drivers.iterate import FixedPointIterator
from openmdao.lib.drivers.newton_solver import NewtonSolver
from openmdao.lib.optproblems import sellar

from openmdao.main.api import Assembly, Component, set_as_top, Driver
from openmdao.main.datatypes.api import Float, Array
from openmdao.main.interfaces import implements, ISolver
from openmdao.main.mpiwrap import MPI
from openmdao.main.test.simpledriver import SimpleDriver
from openmdao.test.execcomp import ExecComp
from openmdao.test.mpiunittest import MPITestCase, MPIContext
from openmdao.util.testutil import assert_rel_error



class ABCDArrayComp(Component):
    delay = Float(0.01, iotype='in')

    def __init__(self, arr_size=10):
        super(ABCDArrayComp, self).__init__()
        self.mpi.requested_cpus = 2

        self.add_trait('a', Array(np.ones(arr_size, float), iotype='in'))
        self.add_trait('b', Array(np.ones(arr_size, float), iotype='in'))
        self.add_trait('c', Array(np.ones(arr_size, float), iotype='out'))
        self.add_trait('d', Array(np.ones(arr_size, float), iotype='out'))

    def execute(self):
        time.sleep(self.delay)
        self.c = self.a + self.b
        self.d = self.a - self.b

    def dump(self, comm):
        print "%d: %s.a = %s" % (comm.rank, self.name, self.a)
        print "%d: %s.b = %s" % (comm.rank, self.name, self.b)
        print "%d: %s.c = %s" % (comm.rank, self.name, self.c)
        print "%d: %s.d = %s" % (comm.rank, self.name, self.d)


class DistribCompSimple(Component):
    """Uses 2 procs but takes full input vars"""
    def __init__(self, arr_size=10):
        super(DistribCompSimple, self).__init__()
        self.mpi.requested_cpus = 2

        self.add_trait('invec', Array(np.ones(arr_size, float), iotype='in'))
        self.add_trait('outvec', Array(np.ones(arr_size, float), iotype='out'))

    def execute(self):
        if self.mpi.comm == MPI.COMM_NULL:
            return
        if self.mpi.comm != MPI.COMM_NULL:
            if self.mpi.comm.rank == 0:
                self.outvec = self.invec * 0.25
            elif self.mpi.comm.rank == 1:
                self.outvec = self.invec * 0.5

            # now combine vecs from different processes
            both = np.zeros((2, len(self.outvec)))
            self.mpi.comm.Allgather(self.outvec, both)

            # add both together to get our output
            self.outvec = both[0,:] + both[1,:]

    def get_req_cpus(self):
        return 2


class DistribInputComp(Component):
    """Uses 2 procs and takes input var slices"""
    def __init__(self, arr_size=11):
        super(DistribInputComp, self).__init__()
        self.arr_size = arr_size
        self.add_trait('invec', Array(np.ones(arr_size, float), iotype='in'))
        self.add_trait('outvec', Array(np.ones(arr_size, float), iotype='out'))

    def execute(self):
        if self.mpi.comm == MPI.COMM_NULL:
            return

        for i,val in enumerate(self.invec):
            self.local_outvec[i] = 2*val

        self.mpi.comm.Allgatherv(self.local_outvec,
                                [self.outvec, self.sizes,
                                 self.offsets, MPI.DOUBLE])

    def get_arg_indices(self, name):
        """ component declares the local sizes and sets initial values
        for all distributed inputs and outputs"""

        comm = self.mpi.comm

        rank = comm.rank
        if name == 'invec':
            base = self.arr_size / comm.size
            leftover = self.arr_size % comm.size
            self.sizes = np.ones(comm.size, dtype="int") * base
            self.sizes[:leftover] += 1 # evenly distribute the remainder across size-leftover procs, instead of giving the whole remainder to one proc

            self.offsets = np.zeros(comm.size, dtype="int")
            self.offsets[1:] = np.cumsum(self.sizes)[:-1]

            start = self.offsets[rank]
            end = start + self.sizes[rank]

            #need to re-initialize the variable to have the correct local size
            self.invec = np.ones(self.sizes[rank], dtype=float)
            self.local_outvec = np.empty(self.sizes[rank], dtype=float)
            return np.arange(start, end, dtype=np.int)

    def get_req_cpus(self):
        return 2

class DistribInputDistribOutputComp(Component):
    """Uses 2 procs and takes input var slices and has output var slices as well"""
    def __init__(self, arr_size=11):
        super(DistribInputDistribOutputComp, self).__init__()
        self.arr_size = arr_size
        self.add_trait('invec', Array(np.ones(arr_size, float), iotype='in'))
        self.add_trait('outvec', Array(np.ones(arr_size, float), iotype='out'))

    def execute(self):
        if self.mpi.comm == MPI.COMM_NULL:
            return

        start = self.offsets[self.mpi.comm.rank]
        for i,val in enumerate(self.invec):
            self.outvec[start+i] = 2*val

    def get_arg_indices(self, name):
        """ component declares the local sizes and sets initial values
        for all distributed inputs and outputs"""

        comm = self.mpi.comm
        rank = comm.rank

        base = self.arr_size / comm.size
        leftover = self.arr_size % comm.size
        self.sizes = np.ones(comm.size, dtype="int") * base
        self.sizes[:leftover] += 1 # evenly distribute the remainder across size-leftover procs, instead of giving the whole remainder to one proc

        self.offsets = np.zeros(comm.size, dtype="int")
        self.offsets[1:] = np.cumsum(self.sizes)[:-1]

        start = self.offsets[rank]
        end = start + self.sizes[rank]

        if name == 'invec':

            #need to re-initialize the variable to have the correct local size
            self.invec = np.ones(self.sizes[rank], dtype=float)
            return np.arange(start, end, dtype=np.int)

        if name == "outvec":
            self.outvec = np.ones(self.sizes[rank], dtype=float)
            return np.arange(start, end, dtype=np.int)

    def get_req_cpus(self):
        return 2

class MPITests1(MPITestCase):

    N_PROCS = 2

    def test_distrib_full_in_out(self):
        size = 11

        top = set_as_top(Assembly())
        top.add("C1", ABCDArrayComp(size))
        top.add("C2", DistribCompSimple(size))
        top.driver.workflow.add(['C1', 'C2'])
        top.connect('C1.c', 'C2.invec')

        top.C1.a = np.ones(size, float) * 3.0
        top.C1.b = np.ones(size, float) * 7.0

        top.run()

        self.assertTrue(all(top.C2.outvec==np.ones(size, float)*7.5))

    def test_distrib_idx_in_full_out(self):
        size = 11

        top = set_as_top(Assembly())
        top.add("C1", ABCDArrayComp(size))
        top.add("C2",DistribInputComp(size))
        top.driver.workflow.add(['C1', 'C2'])
        top.connect('C1.c', 'C2.invec')

        top.C1.a = np.ones(size, float) * 3.0
        top.C1.b = np.ones(size, float) * 7.0

        top.run()

        self.assertTrue(all(top.C2.outvec==np.ones(size, float)*20))


    def test_distrib_idx_in_distrb_idx_out(self):
            size = 11

            top = set_as_top(Assembly())
            top.add("C1", ABCDArrayComp(size))
            top.add("C2",DistribInputDistribOutputComp(size))
            top.add("C3",DistribInputComp(size))
            top.driver.workflow.add(['C1', 'C2', 'C3'])
            top.connect('C1.c', 'C2.invec')
            top.connect('C2.outvec', 'C3.invec')

            top.C1.a = np.ones(size, float) * 3.0
            top.C1.b = np.ones(size, float) * 7.0

            top.run()

            self.assertTrue(all(top.C3.outvec==np.ones(size, float)*40))



if __name__ == '__main__':
    from openmdao.test.mpiunittest import mpirun_tests
    mpirun_tests()