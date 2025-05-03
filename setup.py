from setuptools import find_packages, setup

package_name = 'cbf_polytopes'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fer',
    maintainer_email='lfrecalde1@espe.edu.ec',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'payload_dynamics = cbf_polytopes.payload_dynamics:main',
            'save_bag = cbf_polytopes.save_rosbag:main',
            'read_bag = cbf_polytopes.read_bag:main',
            'read_bag_images = cbf_polytopes.read_bag_images:main',
        ],
    },
)
