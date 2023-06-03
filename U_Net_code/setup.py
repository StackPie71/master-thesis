from setuptools import setup, find_namespace_packages

setup(name='xnet',
      version='1.0.0',
      description='X-Net by Nils Boulanger for Master thesis.',
      author='Nils Boulanger',
      author_email='nils.boulanger@student.uclouvain.be',
      license='',
      install_requires=[
            "torch>=1.6.0a",
            "torchio",
            "scikit-image>=0.14",
            "IPython",
            "numpy",
            "sklearn",
            "nibabel", 'tifffile',
            'json',
            'numpyencoder'
      ],
      py_modules = [],
      entry_points={
          'console_scripts': [],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation']
      )
