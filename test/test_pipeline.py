import io

import dizer3.run_pipeline
import unittest
import os

test_path = os.path.dirname(os.path.realpath(__file__))


def open_test_file(filename: str):
    return io.open(os.path.join(test_path, "files", filename), encoding='utf8')


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dizer3.run_pipeline.process_dir(os.path.join(test_path, "files"), print_files=True)

    def test_parentheticals(self):
        self._validate_rule("parenthetical.txt.segments")

    def test_colon(self):
        self._validate_rule("colon.txt.segments")

    def test_gerund(self):
        self._validate_rule("gerund.txt.segments")

    def test_para_plus_vinf_after_clause_without_v(self):
        self._validate_rule("para + VINF after a clause without V.txt.segments")

    def test_de_forma_a_inf(self):
        self._validate_rule("de forma a + INF.txt.segments")

    def test_adv_plus_gerund(self):
        self._validate_rule("ADV + gerund.txt.segments")

    def test_abbreviations(self):
        self._validate_rule("abbreviations.txt.segments")

    def test_durante(self):
        self._validate_rule("durante.txt.segments")

    def _validate_rule(self, filename):
        return self._validate_files(filename, f"validation.{filename}")

    def _validate_files(self, file_path, validation_file_path):
        with open_test_file(file_path) as file, \
                open_test_file(validation_file_path) as validation_file:
            self.assertListEqual(
                list(validation_file),
                list(file)
            )
