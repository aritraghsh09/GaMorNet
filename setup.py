import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="gamornet-test", # Replace with your own username
	version="0.0.5",
	author="Aritra Ghosh",
	author_email="aritra.ghosh@yale.edu",
	description="A convolutional neural network to separate bulge and disk-dominated galaxies",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/aritraghsh09/GaMorNet",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
	],
	python_requires='>=3.6',
	install_requires=[
		"tensorflow-gpu==1.13.1",
		"tflearn==0.3.2",
		"keras==2.2.4",
		"pandas==1.0.3",
		"wget==3.2",
		"numpy==1.17.0",
		"astropy==3.2.1",
		"matplotlib==3.1.0",
	],

)

