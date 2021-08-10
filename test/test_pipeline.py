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
        with open_test_file("parenthetical.txt.segments") as file, \
                open_test_file("validation.parenthetical.txt.segments") as validation_file:
            self.assertListEqual(
                list(validation_file),
                list(file)
            )

    def test_colon(self):
        with open_test_file("colon.txt.segments") as file, \
                open_test_file("validation.colon.txt.segments") as validation_file:
            self.assertListEqual(
                list(validation_file),
                list(file)
            )
