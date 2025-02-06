# responsible for creating ML project as a package
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOTS = '-e .'
def get_requirements(filename: str) -> List[str]:
    requirements = []
    with open(filename) as file:
        requirements = file.readlines()
        requirements = [requirement.replace('\n', "") for requirement in requirements]

        if(HYPEN_E_DOTS in requirements):
            requirements.remove(HYPEN_E_DOTS)
    return requirements            

setup(
    name = 'AI powered student academic performance prediction',
    version= '0.0.1',
    author= 'Taiyaba Akhtar',
    author_email= 'taibaakhtar42@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt'),

)