# SMARTurinalysis

## Smartphone-based Colorimetric Analysis of Urine Dipsticks for At-Home Prenatal Care

This project is aimed for an automatic detection and evaluation of urine dipsticks from images. Steps performed during the analysis:
* Detection of the stick: Feature matching & Mask R-CNN
* Detection of the reference card: Feature matching
* Localisation of the single test and reference fields 
* Color analysis and comparison

The main scripts to perform the detection and colour evaluation can be found in the **urinalysis** folder.  All scripts that support the main scripts can be found in the **helper** subfolder, scripts analyzing the calculated results in the **evaluation** subfolder. 
The used implementation of Mask R-CNN is in the **mask-rcnn** subfolder. 

## Prerequisites

Required python libraries can be found in requirements.txt

## Related Publication

M. Flaucher et al., "Smartphone-Based Colorimetric Analysis of Urine Test Strips for At-Home Prenatal Care," in IEEE Journal of Translational Engineering in Health and Medicine, vol. 10, pp. 1-9, 2022, Art no. 2800109, doi: 10.1109/JTEHM.2022.3179147.

## Authors

* **Madeleine Flaucher** - *Initial work* 


## Acknowledgments

* [Matterport Inc.](https://github.com/matterport/Mask_RCNN) - Mask R-CNN implementation
