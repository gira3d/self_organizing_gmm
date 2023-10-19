# Self-Organizing Gaussian Mixture Models

Please see the meta-package https://github.com/gira3d/gira3d-reconstruction for detailed documentation.

## Changelog

### New in 0.0.1
- Nanoflann dependency is now external and not pulled using FetchContent
- New CPU implementation of SOGMM container
- New learner class that does not initialize the container repeatedly
- New functions to infer color
- Bugfix to ensure covariance determinant is strictly positive
- Add a mode that runs EM without adaptation (i.e., with a fixed number of components)
