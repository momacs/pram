import setuptools

requires = ['attrs', 'cloudpickle', 'dotmap', 'iteround', 'numpy', 'psutil', 'scipy', 'xxhash']
extras = {
    'vis': ['altair', 'matplotlib', 'selenium', 'PyRQA']
}
extras['all'] = [set(i for j in extras.values() for i in j)]

setuptools.setup(
    name='pypram',
    version='0.1.3',
    author='Tomek D. Loboda',
    author_email='tomek.loboda@gmail.com',
    description='Probabilistic Relational Agent-Based Models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/momacs/pram',
    keywords=['agent-based', 'probabilistic', 'relational', 'model', 'simulation'],
    packages=['pram', 'pram.models'],
    package_dir={'': 'src'},
    python_requires='>=3.6',
    install_requires=requires,
    extras_require=extras,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    license="BSD"
)
