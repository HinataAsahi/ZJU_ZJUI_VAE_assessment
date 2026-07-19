def test_package_imports():
    import vae_project

    assert isinstance(vae_project.__version__, str)
    assert vae_project.__version__ == "0.1.0"
