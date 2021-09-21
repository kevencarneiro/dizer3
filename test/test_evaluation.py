import os
import unittest

import dizer3.evaluation
import dizer3.run_pipeline


class TestEvaluation(unittest.TestCase):
    def test_evaluation(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        evaluation_path = os.path.join(current_path, "evaluation")
        validation_path = os.path.join(evaluation_path, "validation")
        dizer3.run_pipeline.process_dir(evaluation_path, print_files=True)
        dizer3.evaluation.evaluate(evaluation_path, validation_path)
