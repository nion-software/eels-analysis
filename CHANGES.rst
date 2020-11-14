Changelog (nionswift-eels-analysis)
===================================

0.5.2 (2020-11-13):
-------------------
- Fix issue with single channel zero loss peaks in thickness calculation.
- Ensure default background is power law.
- Improve wording in background menu.

0.5.1 (2020-10-06):
-------------------
- Add exponential functions for two-area method.
- Add two-area background fitting class.
- Improve speed of polynomial background and fix power law.

0.5.0 (2020-08-31):
-------------------
- Rework background models to allow for plug-in models.
- Made Align ZLP output the measured shifts in addition to the aligned spectra.
- Allow graphics to select the ROI for aligning spectra in Align ZLP.
- Allow 2D images in Align ZLP and interpret y-axis as sequence axis.
- Reworked Align ZLP to eliminate duplicate code.

0.4.4 (2019-10-24)
------------------
- Changed shift method in ZLP subpixel align to scipy.ndimage.shift with linear interpolation to get rid of artifacts.

0.4.3 (2019-04-29)
------------------
- Added checks to avoid applying live ZLP and thickness to 2D data items.
- Moved ZLP analysis functions to nion.eels_analysis module for better reuse.

0.4.2 (2019-03-27)
------------------
- Make Align ZLP support sequences AND 1D collections.
- Added menu items for center-of-mass and gaussian fit for ZLP alignment (both do sub-pixel alignment but much faster than cross-correlation)

0.4.1 (2019-01-07)
------------------
- Improve data type handling (do not auto-promote to float64).
- Improve menu item layout.

0.4.0 (2018-12-21)
------------------
- Add menu items for live thickness, thickness mapping, align zlp.

0.3.0 (2018-12-12)
------------------
- Nion Swift 0.14 compatibility.
- Use composite line plot display for display again.

0.2.1 (2018-10-14)
------------------
- Update original background subtraction to not use deprecated composite library item.

0.2.0 (2018-09-28)
------------------
- Add simplified background subtraction.
- Add live ZLP tracking.

0.1.1 (2018-05-12)
------------------
- Initial version online.
