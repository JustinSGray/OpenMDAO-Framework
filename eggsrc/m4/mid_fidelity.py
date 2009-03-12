"""
Partial! M4 variable fidelity component.
Consider this only as a hint as to how this might be done.
The code here has only been developed to test feasibility,
and was written by someone without much 'mool' knowledge.
"""

__all__ = ('MidFidelity',)
__version__ = '0.1'

import mool.Optimization.MidFiModel

from openmdao.main import Component, ArrayVariable, Float, Int, String
from openmdao.main.component import RUN_OK, RUN_FAILED
from openmdao.main.variable import INPUT, OUTPUT

import wrapper
import fakes

class MidFidelity(Component):
    """ Wrapper for M4 variable fidelity capability. """

    def __init__(self, name='M4_MidFi', parent=None):
        super(MidFidelity, self).__init__(name, parent)
        self.need_updated_corrections = True

        # Inputs.
        # No 'Option' variables yet.

        String('doe_type', self, INPUT, default='lhs',
               desc='Type of DOE used to generate response surface.')

        String('rs_type', self, INPUT, default='quadratic',
               desc='Type of response surface.')

        Int('n_samples', self, INPUT, default=1, min_limit=1,
            desc='Number of samples.')

        Float('tolerance', self, INPUT, default=1.0e10,
              desc='?')

        Int('correction_function', self, INPUT, default=1,
            desc='Type of correction function.')

        Float('w_h', self, INPUT, default=0.5,
              desc='?')

        Int('accuracy_test_type', self, INPUT, default=2,
            desc='Method for testing accuracy of response.')

        Int('n_samples_test', self, INPUT, default=10, min_limit=1,
            desc='Number of additional samples for additional-points test.')

        Int('ntheta', self, INPUT, default=3,
            desc='For Kriging method, nthets=1(SA),2(Cobyla),3(BFGS)')

        # Sockets.
        fakes.FakeSocket('lofi_model', self, None, 'IComponent', True)
        fakes.FakeSocket('hifi_model', self, None, 'IComponent', True)

        self.input_mappings = []
        self.output_mappings = []

        self._lofi_m4model = None
        self._hifi_m4model = None

        self._midfi_model = mool.Optimization.MidFiModel.Mid_Fi_Model()

        ArrayVariable('sample_points', self, OUTPUT, default=[],
                      desc='Points used to make response',
                      ref_name='sample_points', ref_parent=self._midfi_model)

        ArrayVariable('lofi_results', self, OUTPUT, default=[],
                      desc='Points used to make response',
                      ref_name='lofi_results', ref_parent=self._midfi_model)

        ArrayVariable('hifi_results', self, OUTPUT, default=[],
                      desc='Points used to make response',
                      ref_name='hifi_results', ref_parent=self._midfi_model)

# pylint: disable-msg=E1101
# "Instance of <class> has no <attr> member"

    def set_hifi_model(self, hifi):
        """ Set high fidelity model. """
        self.hifi_model.plugin = hifi
        self._hifi_m4model = None
        self.need_updated_corrections = True

    def set_lofi_model(self, lofi):
        """ Set low fidelity model. """
        self.lofi_model.plugin = lofi
        self._lofi_m4model = None
        self.need_updated_corrections = True

    def add_input_mapping(self, mid, low, high):
        """ Add mapping for input variable. """
        self.input_mappings.append((mid, low, high))
        self.need_updated_corrections = True

    def add_output_mapping(self, mid, low, high):
        """ Add mapping for output variable. """
        self.output_mappings.append((mid, low, high))
        self.need_updated_corrections = True

    def execute(self):
        """ Compute results based on mid-fidelity approximation. """

        # Wrap low fidelity model.
        if self._lofi_m4model is None:
            inputs = []
            for mid, low, high in self.input_mappings:
                inputs.append(low)
            outputs = []
            for mid, low, high in self.output_mappings:
                outputs.append(low)
            self._lofi_m4model = \
                wrapper.M4Model(self.lofi_model.plugin, inputs, outputs)

        # Wrap high fidelity model.
        if self._hifi_m4model is None:
            inputs = []
            for mid, low, high in self.input_mappings:
                inputs.append(high)
            outputs = []
            for mid, low, high in self.output_mappings:
                outputs.append(high)
            self._hifi_m4model = \
                wrapper.M4Model(self.hifi_model.plugin, inputs, outputs)

        # If needed, update correction functions.
        if self.need_updated_corrections:
            nvars = len(self.input_mappings)

            # Check for sufficient DOE samples.
            if self.doe_type == 'ccd':
                # CCD requires an exact value
                if nvars == 1:
                    min_samples = 3
                else:
                    min_samples = 2**nvars + 2*nvars + 1
                if self.n_samples != min_samples:
                    self.warning('Setting n_samples to CCD required value: %d',
                                 min_samples)
                self.n_samples = min_samples
            elif self.doe_type == 'lhs':
                min_samples = nvars
            elif self.doe_type == 'rand_lhs':
                min_samples = nvars
            elif self.doe_type == 'oa2':
                min_samples = (nvars-1)**2
            elif self.doe_type == 'oa3':
                min_samples = (nvars-1)**3
            else:
                self.error("Unknown DOE type '%s'", self.doe_type)
                return RUN_FAILED

            if self.n_samples < min_samples:
                self.warning('Updating n_samples to minimum for DOE: %d',
                             min_samples)
                self.n_samples = min_samples

            # Check for sufficient response surface samples.
            if self.rs_type == 'linear':
                min_samples = nvars + 1
            elif self.rs_type == 'quadratic':
                min_samples = (nvars**2 + 3*nvars + 2)/2
            elif self.rs_type == 'cubic':
                min_samples = (nvars**3 + 6*nvars**2 + 11*nvars + 6) / 6
            elif self.rs_type == 'rbf':
                min_samples = self.n_samples
            elif self.rs_type == 'kriging':
                min_samples = nvars
            else:
                self.error("Unknown RS type '%s'", self.rs_type)
                return RUN_FAILED

            if self.n_samples < min_samples:
                self.warning('Updating n_samples to minimum for RS: %d',
                             min_samples)
                self.n_samples = min_samples

            # Collect upper and lower bounds.
            xlb = []
            xub = []
            for mid, low, high in self.input_mappings:
                xlb.append(self.get(mid+'.min_limit'))
                xub.append(self.get(mid+'.max_limit'))

            self._midfi_model.Set(
                hifi=self._hifi_m4model,
                lofi=self._lofi_m4model,
                xlb=xlb,
                xub=xub,
                nd=nvars,
                nsamples=self.n_samples,
                nsamples_test=self.n_samples_test,
                nf=len(self.output_mappings),
                n_th_func=None,
                accu_test_type=self.accuracy_test_type,
                rs_type=self.rs_type,
                correct_func_type=self.correction_function,
                w_h=self.w_h,
                doe_type=self.doe_type,
                tolerance=self.tolerance,
                ntheta=self.ntheta)

            self._midfi_model.Construct_set_run_type(
                run_type=('doe', 'rs_fast'),
                iflag_doe=3,
                iflag_accu=1)

            self.need_updated_corrections = False

        vec = []
        for mapping in self.input_mappings:
            vec.append(self.get(mapping[0]))

        for i, mapping in enumerate(self.output_mappings):
            self.debug('Running mid-fidelity model %s %d %s',
                       str(vec), i, str(mapping[0]))
            result = self._midfi_model.RunModel(vec, i)
            self.debug('    result %s', str(result))
            setattr(self, mapping[0], result)

        return RUN_OK

