from psai import __version__


def test_smoke_version():
    assert isinstance(__version__, str) and __version__
