from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from config import CONFIG_DIR, get_rule_config, get_search_config


class YamlConfigTests(unittest.TestCase):
    def test_game_yaml_files_exist(self):
        gomoku_path = CONFIG_DIR / "gomoku.yaml"
        go_path = CONFIG_DIR / "go.yaml"
        self.assertTrue(gomoku_path.is_file())
        self.assertTrue(go_path.is_file())

    def test_yaml_shape_and_runtime_config(self):
        for game_name in ("gomoku", "go"):
            path = CONFIG_DIR / f"{game_name}.yaml"
            with path.open("r", encoding="utf-8") as fp:
                payload = yaml.safe_load(fp)
            self.assertIn("rules", payload)
            self.assertIn("search", payload)

            rule = get_rule_config(game_name)
            search = get_search_config(game_name)
            self.assertGreater(rule.board_size, 0)
            self.assertGreater(rule.history_steps, 0)
            self.assertGreater(search.mcts_sims, 0)
            self.assertGreater(search.eval_games, 0)


if __name__ == "__main__":
    unittest.main()
