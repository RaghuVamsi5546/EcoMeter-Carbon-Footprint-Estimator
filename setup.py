from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of requirements.
    """
    requirements = []
    with open(file_path, encoding="utf-8") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="EcoMeter",
    version="0.0.1",
    author="Raghu Vamsi",
    author_email="raghuvamsibolem@gmail.com.com",  
    description="A machine learning project to estimate household carbon footprint based on lifestyle and energy consumption.",
    long_description="Estimate household carbon footprint using ML models.",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
