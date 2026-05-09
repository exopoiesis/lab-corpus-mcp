"""Config resolution tests — adapted from arxiv-radar-mcp/tests/test_config.py.

Same precedence chain as arxiv-radar-mcp:
  explicit arg > $LAB_CORPUS_CONFIG > platformdirs default > ./radar.toml > defaults.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lab_corpus_mcp.config import Config, load


@pytest.fixture
def isolate_config_env(monkeypatch, tmp_path):
    """Make sure no real env var or platformdirs default leaks into the test."""
    monkeypatch.delenv("LAB_CORPUS_CONFIG", raising=False)
    fake_default = tmp_path / "no_such_config.toml"
    monkeypatch.setattr("lab_corpus_mcp.config._default_config_path",
                        lambda: fake_default)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_minimal_toml(p: Path) -> None:
    parsed_dir = (p.parent / "parsed").as_posix()
    cache_dir = (p.parent / "cache").as_posix()
    p.write_text(
        '[embeddings]\n'
        'model = "test-model-name"\n'
        f'cache_dir = "{cache_dir}"\n'
        'batch_size = 8\n'
        '\n'
        '[parse]\n'
        f'dir = "{parsed_dir}"\n'
        '\n'
        '[server]\n'
        'default_k = 7\n',
        encoding="utf-8",
    )


def test_explicit_config_path_is_honoured_when_default_missing(isolate_config_env):
    """Regression mirror of the arxiv-radar-mcp ternary-precedence bug."""
    cfg_file = isolate_config_env / "my-radar.toml"
    _write_minimal_toml(cfg_file)

    cfg = load(cfg_file)

    assert cfg.embeddings.model == "test-model-name"
    assert cfg.embeddings.batch_size == 8
    assert cfg.server.default_k == 7
    assert cfg.parse.dir == isolate_config_env / "parsed"


def test_env_var_used_when_no_explicit_arg(isolate_config_env, monkeypatch):
    cfg_file = isolate_config_env / "env-radar.toml"
    _write_minimal_toml(cfg_file)
    monkeypatch.setenv("LAB_CORPUS_CONFIG", str(cfg_file))

    cfg = load(None)

    assert cfg.embeddings.model == "test-model-name"


def test_falls_back_to_defaults_when_nothing_resolves(isolate_config_env):
    """No explicit, no env, no default file, no cwd radar.toml → built-ins."""
    cfg = load(None)

    assert cfg.embeddings.model == "Qwen/Qwen3-Embedding-4B"
    assert cfg.embeddings.batch_size == 32
    assert cfg.server.default_k == 10
    assert isinstance(cfg.embeddings.cache_dir, Path)
    assert isinstance(cfg.parse.dir, Path)


def test_explicit_arg_wins_over_env_and_default(isolate_config_env, monkeypatch):
    explicit = isolate_config_env / "explicit.toml"
    explicit.write_text(
        '[embeddings]\nmodel = "explicit-wins"\n', encoding="utf-8")
    env_file = isolate_config_env / "env.toml"
    env_file.write_text(
        '[embeddings]\nmodel = "env-loses"\n', encoding="utf-8")
    monkeypatch.setenv("LAB_CORPUS_CONFIG", str(env_file))

    cfg = load(explicit)

    assert cfg.embeddings.model == "explicit-wins"


def test_cwd_radar_toml_picked_up(isolate_config_env):
    """If no explicit / env / platformdirs default, cwd's radar.toml wins."""
    cwd_cfg = isolate_config_env / "radar.toml"
    _write_minimal_toml(cwd_cfg)

    cfg = load(None)

    assert cfg.embeddings.model == "test-model-name"


def test_partial_toml_uses_defaults_for_missing_sections(isolate_config_env):
    """[embeddings] only — [parse] and [server] should still get defaults."""
    cfg_file = isolate_config_env / "partial.toml"
    cfg_file.write_text(
        '[embeddings]\nmodel = "partial"\n', encoding="utf-8")

    cfg = load(cfg_file)

    assert cfg.embeddings.model == "partial"
    assert cfg.server.default_k == 10  # default
    # parse.dir defaults to <embeddings.cache_dir.parent>/parsed
    assert cfg.parse.dir == cfg.embeddings.cache_dir.parent / "parsed"


def test_defaults_returns_independent_instances():
    """Config.defaults() must not share mutable state across calls."""
    a = Config.defaults()
    b = Config.defaults()
    assert a is not b
    assert a.embeddings is not b.embeddings


def test_default_config_path_returns_platform_path():
    """Smoke: _default_config_path() must yield a concrete Path under the
    platform user-config dir (covers the otherwise-monkeypatched line)."""
    from lab_corpus_mcp.config import _default_config_path

    p = _default_config_path()
    assert isinstance(p, Path)
    assert p.name == "radar.toml"


def test_platformdirs_default_used_when_present(monkeypatch, tmp_path):
    """When no explicit / env arg, but the platformdirs default file
    exists on disk, it gets loaded."""
    monkeypatch.delenv("LAB_CORPUS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    fake_default = tmp_path / "platform-default.toml"
    fake_default.write_text(
        '[embeddings]\nmodel = "from-platform-default"\n', encoding="utf-8")
    monkeypatch.setattr("lab_corpus_mcp.config._default_config_path",
                        lambda: fake_default)

    cfg = load(None)
    assert cfg.embeddings.model == "from-platform-default"
