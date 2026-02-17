from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from config import (
    CONFIG_DIR,
    DEVICE,
    GAME_NAME,
    get_logging_config,
    get_model_config,
    get_rule_config,
    get_runtime_config,
    get_search_config,
    get_system_config,
    get_train_config,
)


class YamlConfigTests(unittest.TestCase):
    def test_game_yaml_files_exist(self):
        gomoku_path = CONFIG_DIR / "gomoku.yaml"
        go_path = CONFIG_DIR / "go.yaml"
        system_path = CONFIG_DIR / "system.yaml"
        self.assertTrue(gomoku_path.is_file())
        self.assertTrue(go_path.is_file())
        self.assertTrue(system_path.is_file())

    def test_yaml_shape_and_runtime_config(self):
        for game_name in ("gomoku", "go"):
            path = CONFIG_DIR / f"{game_name}.yaml"
            with path.open("r", encoding="utf-8") as fp:
                payload = yaml.safe_load(fp)
            self.assertIn("rules", payload)
            self.assertIn("search", payload)
            self.assertIn("model", payload)
            self.assertIn("train", payload)
            self.assertIn("runtime", payload)
            self.assertIn("logging", payload)

            rule = get_rule_config(game_name)
            search = get_search_config(game_name)
            model = get_model_config(game_name)
            train = get_train_config(game_name)
            runtime = get_runtime_config(game_name)
            logging_cfg = get_logging_config(game_name)
            self.assertGreater(rule.board_size, 0)
            self.assertGreater(rule.history_steps, 0)
            self.assertGreater(search.mcts_sims, 0)
            self.assertGreater(search.eval_games, 0)
            self.assertGreater(model.channels, 0)
            self.assertGreater(model.num_res, 0)
            self.assertGreater(train.buffer_size, 0)
            self.assertGreater(train.batch_size, 0)
            self.assertGreater(train.train_updates, 0)
            self.assertGreater(train.learning_rate, 0)
            self.assertGreater(train.weight_decay, 0)
            self.assertGreater(train.selfplay_games, 0)
            self.assertTrue(runtime.model_dir)
            self.assertTrue(runtime.data_dir)
            self.assertTrue(runtime.log_dir)
            self.assertTrue(runtime.tb_dir)
            self.assertTrue(logging_cfg.level)
            self.assertTrue(logging_cfg.name)

    def test_system_yaml_shape_and_runtime_config(self):
        system_path = CONFIG_DIR / "system.yaml"
        with system_path.open("r", encoding="utf-8") as fp:
            payload = yaml.safe_load(fp)
        self.assertIn("system", payload)
        self.assertIn("default_game", payload["system"])
        self.assertIn("device", payload["system"])

        system_cfg = get_system_config()
        self.assertTrue(system_cfg.default_game)
        self.assertTrue(system_cfg.device)
        self.assertEqual(system_cfg.default_game, GAME_NAME)
        self.assertTrue(DEVICE)


if __name__ == "__main__":
    unittest.main()
