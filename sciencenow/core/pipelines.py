from abc import ABC
from typing import List

from sciencenow.core.steps import Step

class Pipeline(ABC):
    """
    Abstract Base Class for all pre- and postprocessing Pipelines.
    """
    steps: List[Step]

    def execute(self, input):
        """
        Cycle through an arbitrary number of steps until the output has the desired form.
        Then return output.
        """
        for index, step in enumerate(self.steps):
            if index == 0:
                output = step.execute(input)
            else:
                output = step.execute(output)
        return output


class PubmedPipeline(Pipeline):
    def __init__(self, steps:List[Step]) -> None:
        super().__init__()
        self.steps = steps


class ArxivPipeline(Pipeline):
    def __init__(self, steps:List[Step]) -> None:
        super().__init__()
        self.steps = steps
