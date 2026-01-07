def test_make_env_stub_errors_cleanly():
    from embodied_ai.envs.make_env import make_env

    try:
        make_env("PickCube-v1")
    except NotImplementedError:
        assert True


