import unittest
import keybindHandler


class ControlScheme(unittest.TestCase):
    transform = keybindHandler.control_scheme_transform(".\\Resources\\testfiles\\configa.json", ".\\Resources\\testfiles\\configb.json")
    def test_config(self):
        a_oh_input = [0,0,0,0,0,0,0]
        b_oh_expected = [0,0,0,0,0,0,0]
        self.assertTrue(ControlScheme.transform(a_oh_input) == b_oh_expected)

    def test_config_2(self):
        a_oh_input = [1,1,0,0,0,0,0]
        b_oh_expected = [1,1,0,0,0,0,0]
        self.assertTrue(ControlScheme.transform(a_oh_input) == b_oh_expected)


if __name__ == "__main__":
    unittest.main()
