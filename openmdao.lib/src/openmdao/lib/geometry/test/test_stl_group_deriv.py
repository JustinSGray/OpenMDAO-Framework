"""
Testing differentiation of stl group objects.
"""


import os
import unittest

import numpy as np

from openmdao.lib.components.geomcomp import GeomComponent
from openmdao.main.api import Component, Assembly, set_as_top
from openmdao.main.datatypes.api import Float, Array
from openmdao.main.interfaces import IParametricGeometry, implements, \
                                     IStaticGeometry
from openmdao.main.variable import Variable
from openmdao.util.testutil import assert_rel_error


from openmdao.lib.geometry.stl_group import STLGroup
import openmdao.lib.geometry.stl as stl
from openmdao.lib.geometry.ffd_axisymetric import Body, Shell
from openmdao.lib.geometry.stl_group import STLGroup

import openmdao.examples.nozzle_geometry_doe


class GeomRecieveDerivApplyDeriv(Component): 
    """Takes an STLGroup object in and outputs an nx3 array of points from that 
    STL Group"""

    geom_in = Variable(iotype='in')
    out = Array(iotype='out')

    def execute(self): 
        self.out = self.geom_in

    def linearize(self): 
        
        self.J = np.eye(self.data_size)

    def apply_deriv(self, arg, result):
        if 'geom_in' in arg:
            result['out'] += self.J.dot(arg['geom_in'])
    
    def apply_derivT(self, arg, result):
        if 'out' in arg:
            result['geom_in'] += self.J.T.dot(arg['out'])

class PlugNozzleGeometry(STLGroup): 

    def __init__(self): 
        super(PlugNozzleGeometry,self).__init__()

        this_dir, this_filename = os.path.split(os.path.abspath(openmdao.examples.nozzle_geometry_doe.__file__))
        plug_file = os.path.join(this_dir, 'plug.stl')
        plug = stl.STL(plug_file)
        cowl_file = os.path.join(this_dir, 'cowl.stl')
        cowl = stl.STL(cowl_file)
        
        n_c = 10
        body = Body(plug,controls=n_c) #just makes n_C evenly spaced points
        shell = Shell(cowl,cowl.copy(),n_c,n_c)

        self.add(body,name="plug")
        self.add(shell,name="cowl")

class DummyGeometry(object): 
    implements(IParametricGeometry, IStaticGeometry)

    def __init__(self): 
        self.vars = {'x':np.array([1,2]), 'y':1, 'z':np.array([0,0])}
        self._callbacks = []


    def list_parameters(self): 
        self.params = []
        meta = {'value':np.array([1,2]), 'iotype':'in', 'shape':(2,)}
        self.params.append(('x', meta))

        meta = {'value':1.0, 'iotype':'in',}
        self.params.append(('y', meta))

        meta = {'value':np.array([0,0]), 'iotype':'out', 'shape':(2,)}
        self.params.append(('z', meta))

        meta = {'iotype':'out', 'data_shape':(2,), 'type':IStaticGeometry}
        self.params.append(('geom_out',meta))

        return self.params

    def set_parameter(self, name, val): 
        self.vars[name] = val

    def get_parameters(self, names): 
        return [self.vars[n] for n in names]

    def linearize(self): 
        self.J = np.array([[2, 0, 1],
                           [0, 2, 1]])
        self.JT = self.J.T

    def apply_deriv(self, arg, result): 
        if 'x' in arg: 
            if 'z' in result: 
                result['z'] += self.J[:,:2].dot(arg['x'])
            if 'geom_out' in result:
                result['geom_out'] += self.J[:,:2].dot(arg['x'])
        if 'y' in arg: 
            if 'z' in result: 
                result['z'] += self.J[:,2]*arg['y']
            if 'geom_out' in result:
                result['geom_out'] += self.J[:,2]*arg['y']

        return result


    def apply_derivT(self, arg, result): 
        if 'z' in arg: 
            if 'x' in result:
                result['x'] += self.JT[:2,:].dot(arg['z'])
            if 'y' in result: 
                result['y'] += self.JT[2,:].dot(arg['z'])
        if 'geom_out' in arg: 
            if 'x' in result:
                result['x'] += self.JT[:2,:].dot(arg['geom_out'])
            if 'y' in result: 
                result['y'] += self.JT[2,:].dot(arg['geom_out'])
        
        return result


    def regen_model(self):
        x = self.vars['x']
        y = self.vars['y']

        self.z = 2*x + y 
        self.vars['z'] = self.z

    def get_static_geometry(self): 
        return self

    def register_param_list_changedCB(self, callback):
        self._callbacks.append(callback)

    def _invoke_callbacks(self): 
        for cb in self._callbacks: 
            cb()

    def get_visualization_data(self, wv): #stub
        pass


class TestcaseDerivSTLGroup(unittest.TestCase):

    def setUp(self): 
        self.top = set_as_top(Assembly())
        self.top.add('geom', GeomComponent())
        self.top.geom.add('parametric_geometry', PlugNozzleGeometry())

        self.top.add('rec', GeomRecieveDerivApplyDeriv())

        self.top.connect('geom.geom_out', 'rec.geom_in')

    def test_deriv(self): 
        print self.top.geom.list_inputs()

if __name__ == "__main__": 

    unittest.main()
