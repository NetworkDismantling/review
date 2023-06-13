from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install


def custom_command():
    # pass

    from subprocess import check_output

    folder = 'network_dismantling/common/external_dismantlers/'
    cd_cmd = 'cd {} && '.format(folder)
    cmd = 'make clean && make'

    try:
        print(check_output(cd_cmd + cmd, shell=True, text=True))
    except Exception as e:
        print("ERROR! {}".format(e))

        exit(-1)


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        custom_command()


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        custom_command()


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        custom_command()


setup(
    name='NetworkDismantling',
    version='0.1',
    packages=find_packages(),
    url='',
    license='',
    author='Marco Grassia',
    author_email='',
    description='0.1',
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
        'egg_info': CustomEggInfoCommand,
    }
)
