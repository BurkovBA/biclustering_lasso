import os

from distutils.core import setup, Extension, Command
import numpy as np


def main():
    """
    During development compile and install this extension with:

    python setup.py clean --all build --debug install --record files.txt
    """
    biclustering_lasso_extension = Extension(
        "biclustering_lasso",
        ["biclustering_lasso/biclustering_lasso.c"],
        include_dirs=[os.path.join(np.get_include(), 'numpy')]
    )

    setup(
        name="biclustering_lasso",
        version="1.0.0",
        description="Python interface for biclustering algorithm, based on quadratic programming",
        author="Boris A. Burkov",
        author_email="vasjaforutube@gmail.com",
        ext_modules=[
            biclustering_lasso_extension,
            # Extension("hello", ["biclustering/hello.c"])
        ],
        cmdclass={
            'uninstall': Uninstall,
            'test': Test,
        },
    )

# Custom commands
# ---------------

# For a tutorial on custom commands see: https://dankeder.com/posts/adding-custom-commands-to-setup-py/


class Uninstall(Command):
    """Removes the wheel and .so files from the site-packages directory."""
    description = 'Removes the wheel and .so files from the site-packages directory.'

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for line in open("files.txt"):
            os.system(f"rm {line}")


class Test(Command):
    """Runs unit-tests."""
    description = 'Runs unit-tests.'

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system(f"python -m unittest tests/test_biclustering_lasso.py")


if __name__ == "__main__":
    main()
