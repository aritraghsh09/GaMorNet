# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added

### Changed

### Removed

### Fixed


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
