from setuptools import setup, find_packages
from glob import glob

package_name = "cyberrunner_state_estimation"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test/"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, glob("calib/*.txt")),
        #("share/" + package_name, "rviz/config.rviz"),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="timflueckiger",
    maintainer_email="timflueckiger@outlook.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "estimator = cyberrunner_state_estimation.cyberrunner_state_estimation_node:main",
            "estimator_sub = cyberrunner_state_estimation.cyberrunner_state_estimation_subimg:main",
            "select_markers = cyberrunner_state_estimation.select_markers:main",
        ],
    },
)
