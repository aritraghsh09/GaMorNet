import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="gamornet-cpu", # Replace with your own username
	version="0.2.3",
	author="Aritra Ghosh",
	author_email="aritraghsh09@gmail.com",
	description="A CNN to classify galaxies morphologically",
	long_description=long_description,
	long_description_content_type="text/markdown",
	keywords="astrophysics astronomy galaxies convolutional neural networks morphological analysis morphology sdss candels",
	url="http://gamornet.ghosharitra.com/",
	project_urls={
		"Source Code": "https://github.com/aritraghsh09/GaMorNet",
		"Documentation": "https://gamornet.readthedocs.io/",
	},
	classifiers=[
		"Programming Language :: Python :: 3",
		"Development Status :: 4 - Beta",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering :: Astronomy",
		"Topic :: Scientific/Engineering :: Physics",
	],
	packages=setuptools.find_packages(),
	python_requires='~=3.3',
	install_requires=[
		"tensorflow-cpu ~=1.12",
		"tflearn ~=0.3",
		"keras ~=2.2",
		"wget >=3.2",
		"numpy >=1.16",
		"progressbar2",
	],
)

