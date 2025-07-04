from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str) -> List[str]:
    """
    this function will return list of requirements"""
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','')for req in requirements]

    if  HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)


    return requirements    

setup(
    name="EcoMeter",
    version="0.0.1",
    author="Raghu Vamsi",
    author_email="raghuvamsibolem.com",
    description="A machine learning project to estimate household carbon footprint based on lifestyle and energy consumption.",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)