from pathlib import Path
import pandas as pd
from typing import Generator
from .hl7_parser import HL7


class HL7Library:
    """Stores HL7 library as a pandas dataframe"""

    def __init__(self, root: Path):
        self.data = HL7Library.retrieve_data(root)

    @staticmethod
    def read_hl7(filepath: Path) -> str:
        "Read all text from a file"
        with open(filepath, "r") as f:
            text = f.read()
        return text

    @staticmethod
    def get_files(root: Path) -> Generator[Path, None, None]:
        """Gets all files under the root data folder
        Exclude files starting with "All" as we are only interested
        in individual files. Also exclude directories
        """
        return (f for f in root.glob("**/[!All]*") if not f.is_dir())

    @staticmethod
    def create_df(files: Generator[Path, None, None]) -> pd.DataFrame:
        "Return dataframe generated from all HL7 files"
        data = []

        for f in files:
            patient = HL7(HL7Library.read_hl7(f))
            data.append(
                {
                    "Name": patient.name,
                    "Sex": patient.sex,
                    "DOB": patient.dob,
                    "Source": str(f),  # only for debug purposes
                }
            )

        return pd.DataFrame(data)

    @staticmethod
    def retrieve_data(root: Path) -> pd.DataFrame:
        "Gets all valid files and return dataframe"
        files = HL7Library.get_files(root)
        patient_data = HL7Library.create_df(files)
        return patient_data


if __name__ == "__main__":
    # root data directory
    data_dir = Path("./HW2/Clinical_HL7_Samples")
    patient_data = HL7Library(data_dir).data

    # for debugging purposes
    patient_data.to_csv("patients.csv")
    print(patient_data.head())
    assert patient_data.Sex.all() in ("M", "F"), "Invalid Sex found in data"

    # Answers
    valid_dobs = patient_data[patient_data.DOB.notnull()]
    print(
        f"The youngest male patient was born on {max(valid_dobs.DOB[valid_dobs.Sex =='M'])}"
    )
    print(
        f"There are {len(set(patient_data.Name[patient_data.Sex == 'F']))} unique females"
    )
