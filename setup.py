from glob import glob

from setuptools import find_packages, setup


package_name = "rus_wm"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/ui", glob("ui/*.ui")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Feng Li",
    maintainer_email="feng.li@tum.de",
    description="DeepUSNav ROS 2 operator console for ultrasound-guided FR3 experiments.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "deepusnav_console = rus_wm.deepusnav_console:main",
            "deepusnav_inference = rus_wm.deepusnav_inference:main",
            "deepusnav_jog_adapter = rus_wm.jog_adapter:main",
        ],
    },
)
