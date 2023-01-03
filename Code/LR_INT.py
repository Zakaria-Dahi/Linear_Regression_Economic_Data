import abc
class LR_INT (abc.ABC):

    """
    About: This is an interface to build up a design pattern
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def linear_regression(self,*args):
        pass

    @abc.abstractmethod
    def display_result(self,*args):
        pass

