BUILD_TEST=0 python setup.py bdist_wheel
pip install ./dist/torch-2.1.0a0+gitbe5a80b-cp310-cp310-linux_x86_64.whl --force-reinstall --no-dependencies
