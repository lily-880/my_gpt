"""Day 1：确认 pytest 能发现测试。Day 2 起换成真正的模型和 mask 测试。"""


def test_package_importable():
    import my_gpt

    assert my_gpt.__version__ == "0.1.0"
