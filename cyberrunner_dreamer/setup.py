from setuptools import setup, find_packages

package_name = "cyberrunner_dreamer"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test/"]),
    # package_data={"cyberrunner_dreamer": ["data/*.txt"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["data/path_0002_hard.pkl"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Thomas Bi",
    maintainer_email="bit@ethz.ch",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "train = cyberrunner_dreamer.train:main",
            "train_parallel = cyberrunner_dreamer.train_parallel:main",
            "test = cyberrunner_dreamer.test_motors:main",
            "eval = cyberrunner_dreamer.eval:main",
        ],
    },
)
