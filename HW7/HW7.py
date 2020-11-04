from __future__ import annotations
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pydicom


@dataclass
class Point:
    x: int
    y: int

    def distance_from(self, p2: Point, scale: Point = None) -> float:
        "Calculate distance between two points scaled by scale"
        return (
            (((self.x - p2.x) * scale.x) ** 2 + ((self.y - p2.y) * scale.y) ** 2)
        ) ** 0.5


def hounsfield_scale(pxls: np.ndarray, scale: float, intercept: float) -> float:
    """Pixel intensities are converted into HUs using the
    following simple formula

        HU = (Pixel Intensity Value) × RescaleSlope + RescaleIntercept
    """
    return pxls * scale + intercept


def main():

    # Read in data
    this_dir = Path(__file__).parent.absolute()
    with open(this_dir / Path("data/test_CT"), "rb") as infile:
        ds = pydicom.dcmread(infile)

    # Question 7
    """
    Using text_CT file provided, which of the following is true about the patient this image belongs to?

    Female, born in 1969
    Male, born in 1970
    Male, born in 1969
    Female, born in 1970
    """
    logger = logging.getLogger("Question 7")
    data_elements = ["PatientName", "PatientSex", "PatientBirthDate"]

    for de in data_elements:
        logger.info(ds.data_element(de))

    # Question 8
    """
    One of the advantages of DICOM files is that they can be calibrated to real physical scales. For instance, you can measure distances on DICOM images in millimeters, thus corresponding to the true, real sizes.
    Open the test_CT file that I have provided; it is a 512x512 CT image.  What is the distance between pixels A=(151,30) and B=(200,300)?

    15 mm
    185 mm
    285 mm
    385 mm
    485 mm
    """
    logger = logging.getLogger("Question 8")
    logger.debug(f"Image size.......: {ds.pixel_array.shape}")
    A = Point(151, 30)
    B = Point(200, 300)
    scale = Point(*ds.PixelSpacing)
    logger.info(
        f"Distance between {A} and {B} scaled by {scale} is {A.distance_from(B, scale=scale)}"
    )

    # Question 9
    """ 
    The idea of calibration can be extended to the image pixels as well. Pixel intensities in most image formats are unitless – that is, they are not usually mapped into any physical scale. However, in medical CT imaging pixel values can be converted to the universal Hounsfield units (HU), directly corresponding to various tissue densities found in human body (soft tissues, bones, lungs, brain, etc; see https://en.wikipedia.org/wiki/Hounsfield_scale) 

    Pixel intensities are converted into HUs using the following simple formula

    HU = (Pixel Intensity Value) × RescaleSlope + RescaleIntercept

    Using test_CT image provided with this homework, do the following:
    Read the image with pydicom
    Extract pixel values as pixel_array (2D list)
    Find RescaleIntercept and RescaleSlope values in its DICOM tags
    Using the formula above, convert all image intensities into HU units

    Now, assuming that bone HU range corresponds to HU>=300, how many bone pixels we have in this image?
    (as always, pick the answer closest to yours)

    1700
    2700
    3700
    4700
    5700
    """
    logger = logging.getLogger("Question 9")
    logger.debug(
        f"Rescale Slope: {ds.RescaleSlope}, Rescale Intercept {ds.RescaleIntercept}"
    )
    HU = hounsfield_scale(ds.pixel_array, ds.RescaleSlope, ds.RescaleIntercept)
    logger.info(f"Bone Pixels : {np.sum(HU>300)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
