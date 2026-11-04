from nao_motion.config.env import apply_env, parse_env_file


def test_env_file_overrides_robot_yaml(tmp_path, cfg):
    f = tmp_path / ".env"
    f.write_text("# commentaire\nNAO_IP=10.0.0.7   # ip\nNAOQI_VERSION=2.1.4.13\nNAO_PORT=9559\nNAO_NAME=nao\n")
    assert parse_env_file(f)["NAO_IP"] == "10.0.0.7"
    r = apply_env(cfg.robot, f, environ={"NAO_PORT": "9600"})
    assert r["robot"]["ip"] == "10.0.0.7" and r["robot"]["naoqi_version"] == "2.1.4.13"
    assert r["robot"]["port"] == 9600  # l'environnement l'emporte sur .env
    assert "NAO_IP" in r["_env_sources"]


def test_example_env_is_parseable(cfg):
    vals = parse_env_file(cfg.root.parent / ".env.example")
    assert vals["NAOQI_VERSION"] == "2.1.4.13" and vals["NAO_MODEL"] == "naoH25"
    r = apply_env(cfg.robot, cfg.root.parent / ".env.example", environ={})
    assert r["robot"]["name"] == "nao" and r["bridge"]["port"] == 9600
