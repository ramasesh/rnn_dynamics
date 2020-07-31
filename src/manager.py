class MeasurementManager:

  def __init__(self, static_state, reporter):
    self.static_state = static_state
    self.reporter = reporter
    self.measurements = {}

  def add_measurement(self, measurement_spec):
    name = measurement_spec['name']
    self.measurements[name] = {'interval': measurement_spec['interval'],
                               'function': measurement_spec['function']}

  def process(self, step, dynamic_state):
    """ decides which msmts need to be performed at this step
    and performs them """

    to_measure = []
    for msmt_name, msmt_spec in self.measurements.items():
      if step % msmt_spec['interval'] == 0:
        to_measure.append(msmt_name)

    if len(to_measure) > 0:
      self.trigger_subset(step, dynamic_state, to_measure)

  def trigger_subset(self, step, dynamic_state, msmt_list):
    """ performs the measurements specified by measurement_list """
    full_state = {**self.static_state,
                  **dynamic_state}

    step_msmts = {}
    for msmt_name in msmt_list:
      msmt_fun = self.measurements[msmt_name]['function']
      msmt_value = msmt_fun(full_state)

      step_msmts[msmt_name] = msmt_value

    self.reporter.report_all(step, step_msmts)
