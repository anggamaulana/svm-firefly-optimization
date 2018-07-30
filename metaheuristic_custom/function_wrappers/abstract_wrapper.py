# In order to make this class AbstractBaseClass: 
from abc import ABCMeta, abstractmethod

class AbstractWrapper(object):

    # Return value: Array
    @abstractmethod
    def maximum_decision_variable_values(self):
      pass

    # Return value: Array
    @abstractmethod
    def minimum_decision_variable_values(self):
      pass 

    # Input value: Array
    @abstractmethod
    def objective_function_value(self, decision_variable_values):
      pass

    # For the algorithm that requires initial estimate that is depending on the particular objective function:
    # Return value: Array
    @abstractmethod
    def initial_decision_variable_value_estimates(self):
      pass

    @abstractmethod
    def logging(self,report,datas):
      pass

    @abstractmethod
    def report(self):
      pass

    @abstractmethod
    def reportexcel(self):
      pass