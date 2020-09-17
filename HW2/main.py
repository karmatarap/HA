from pathlib import Path
import pandas as pd
import logging
from typing import Generator
from .hl7_parser import hl7_to_dict, extract_sex, extract_dob


def read_hl7(filepath: Path) -> str:
    "Read all text from a file"
    with open(filepath, "r") as f:
        text = f.read()
    return text


def get_files(root: Path) -> Generator[Path, None, None]:
    """Gets all files under the Clinical data folder
    Exclude files starting with "All" as we are only interested
    in individual files. Also exclude directories
    """
    return (f for f in root.glob("**/[!All]*") if not f.is_dir())


def create_df(files: Generator[Path, None, None]) -> pd.DataFrame:
    "Return dataframe generated from all HL7 files"
    data = []

    for f in files:
        patient_dict = hl7_to_dict(read_hl7(f))
        data.append(
            {
                "Sex": extract_sex(patient_dict),
                "DOB": extract_dob(patient_dict),
                "Source": str(f),
            }
        )

    return pd.DataFrame(data)


def retrieve_data(root: Path) -> pd.DataFrame:
    "Gets all valid files and return dataframe"
    files = get_files(root)
    patient_data = create_df(files)
    return patient_data


if __name__ == "__main__":
    # root data directory
    data_dir = Path("./HW2/Clinical_HL7_Samples")
    patients = retrieve_data(data_dir)
    patients.to_csv("patients.csv")
    print(patients.head())
    assert patients.Sex.all() in ("M", "F"), "Invalid Sex found in data"
    print(
        f"The youngest male patient was born on {max(patients.DOB[patients.Sex =='M']).isoformat()}"
    )
    print(f"There are {len(patients[patients.Sex == 'F'])} unique females")
