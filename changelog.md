# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

## [Unreleased]

### Added

### Changed

### Removed

### Fixed

## [0.4.2] -- 2020-07-14

### Changed/Fixed
- Changed setup.py to force Keras installation of <2.4 

## [0.4.1] -- 2020-06-14

### Changed/Fixed
- Corrected errors in the Python Docstrings of both keras and tflearn modules
- Corrected errors/typos in API documentation 


## [0.4] -- 2020-06-12

### Added
- Documentation and Tutorials
- Code is now Production Ready as Well



## [0.3.1] - 2020-06-10

### Fixed
- gamornet_predict_tflearn can handle the situation now when batch_size > number of images



## [0.3] - 2020-06-04

### Added
- Basic Documentation for the Project

### Changed
- updated LRN function in keras_module to remove dependancy on image_dim_ordering function of keras backend
- altered the naming scheme of saved tflearn models.
- increased the flushing time for the gamornet_predict_tflearn function progressbar
- implemented single version and author sourcing. (both are sourced from __init__.py to setup.py


## [0.2.1] - 2020-05-15
### Fixed
- the tflearn_module now properly imports functions from the keras_module

## [0.2] - 2020-05-15
### Added
- tflearn_module

### Changed
- Alterations to some of the inner working of the keras_module user facing functions.


## [0.1] - 2020-05-12
### Added
- Initial Release of GaMorNet
