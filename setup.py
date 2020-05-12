import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="gamornet-test", # Replace with your own username
	version="0.1",
	author="Aritra Ghosh",
	author_email="aritra.ghosh@yale.edu",
	description="A CNN to classify galaxies morphologically",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="http://gamornet.ghosharitra.com/",
	project_urls={
		"Source Code": "https://github.com/aritraghsh09/GaMorNet",
		"Documentation": "https://gamornet.readthedocs.io/",
	},
	classifiers=[
		"Programming Language :: Python :: 3",
		"Development Status :: 4 - Beta",
		"Environment :: GPU :: NVIDIA CUDA :: 10.1",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering :: Astronomy",
		"Topic :: Scientific/Engineering :: Physics",
	],
	packages=setuptools.find_packages(),
	python_requires='==3.6',
	install_requires=[
		"tensorflow-gpu==1.15.2",
		"tflearn==0.3.2",
		"keras==2.2.5",
		"pandas==1.0.3",
		"wget==3.2",
		"numpy==1.16.4",
		"astropy==3.2.1",
		"matplotlib==3.1.0",
	],

)

