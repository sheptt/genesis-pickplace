"""Checks for data leakage, temporal alignment and false-positive task scores."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from robot_lab.dataset import load_dataset, split_episodes, training_arrays
from robot_lab.policy import load_policy, Policy
from robot_lab.simulation import assess_success
import torch


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes, _ = load_dataset()

    def test_episode_splits_have_no_leakage(self):
        splits = split_episodes(self.episodes)
        sets = [set(e['id'] for e in group) for group in splits.values()]
        self.assertEqual(len(set.union(*sets)), len(self.episodes))
        for i in range(3):
            for j in range(i):
                self.assertFalse(sets[i] & sets[j])

    def test_first_label_is_next_action(self):
        ep = self.episodes[0]
        x, pos, closed, force = training_arrays([ep])
        self.assertEqual(len(x), 157)
        np.testing.assert_array_equal(pos, ep['action'][1:, :3])
        np.testing.assert_array_equal(force, ep['action'][1:, 7:8])

    def test_checkpoint_reload_preserves_predictions(self):
        model = Policy().eval()
        ep = self.episodes[0]
        expected = model.predict(ep['obj_xy'], ep['magnitude'], [.2,.5,.9])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'weights.pt'
            torch.save(model.state_dict(), path)
            other = Policy()
            other.load_state_dict(torch.load(path, weights_only=True))
            actual = other.predict(ep['obj_xy'], ep['magnitude'], [.2,.5,.9])
        np.testing.assert_array_equal(expected, actual)

    def test_sliding_and_unreleased_grasp_do_not_count(self):
        slide = [{'cube_xyz':[.46,0.,.04]} for _ in range(60)]
        self.assertFalse(assess_success(slide, 0.)['success'])
        lifted = [{'cube_xyz':[.64,0.,.2]}]+slide
        self.assertFalse(assess_success(lifted, .5)['success'])
        self.assertTrue(assess_success(lifted, 0.)['success'])


if __name__ == '__main__':
    unittest.main()
