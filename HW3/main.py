from pathlib import Path
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class FacilityTimes:
    "Models a Hospital Facility with arrival, exam start and end times"

    def __init__(
        self, df: pd.DataFrame, start_date_col: str, end_date_col: str
    ) -> None:
        self._validate(df)  # defensive coding
        self.data = FacilityTimes.add_date_features(df)  # add new features
        self.patient_col = "stMRN"
        self.start_date_col = start_date_col
        self.end_date_col = end_date_col
        self.date_interval = self._date_interval()
        self.patient_counts = self.patient_interval_counts()

    @staticmethod
    def _validate(df) -> None:
        req_cols = ("stMRN", "dtArrive", "dtBegin", "dtCompleted")
        if not all(col in df.columns for col in req_cols):
            raise AttributeError(
                f"Must have the following columns:{','.join(req_cols)}"
            )

    @staticmethod
    def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add Date Features of interest: 
        Add Durations between visits
        Split datetimes to date and time portions
        """
        df["waitTime"] = df["dtBegin"] - df["dtArrive"]
        df["examTime"] = df["dtCompleted"] - df["dtBegin"]
        df["totalTime"] = df["dtCompleted"] - df["dtArrive"]
        return df

    def count_patients_at_range(self, hour_start: int, hour_end: int) -> int:
        """ Returns count of patients within provided range

        For ranges A->B and C->D, overlapping intervals
            must have property: Not(B<=C or A>=D)
        Let A = hour_start, B = hour_end
        Let C = start_date_col, D = end_date_col
        """
        overlaps = self.data[
            ~(
                (time(hour_end) <= self.data[self.start_date_col].dt.time)
                | (time(hour_start) >= self.data[self.end_date_col].dt.time)
            )
        ]

        return len(overlaps[self.patient_col].unique())

    def _subset_by_date(self, date: datetime) -> pd.DataFrame:
        "Returns a subset version of the current dataset where a given date exists in the date range of interest"
        return self.data[
            (self.data[self.start_date_col] <= date)
            & (date <= self.data[self.end_date_col])
        ]

    def _date_interval(
        self, freq: str = "30min", start_time: int = 7, end_time: int = 20
    ) -> pd.date_range:
        "Returns date range for fixed interval"
        return pd.date_range(
            start=self.data[self.start_date_col].dt.date.min()
            + timedelta(hours=start_time),
            end=self.data[self.end_date_col].dt.date.max() + timedelta(hours=end_time),
            freq=freq,
        )

    def _count_unique_patients_at(self, date: datetime) -> int:
        "Returns count of patients at a timepoint"
        return len(self._subset_by_date(date)[self.patient_col].unique())

    def patient_interval_counts(self) -> np.array:
        "Returns patient counts at each interval provided"
        return np.array(list(map(self._count_unique_patients_at, self.date_interval)))

    def max_patient_count(self) -> int:
        "Max count of patients in an interval"
        return self.patient_counts.max()

    def max_interval(self) -> pd.date_range:
        "Interval at which max patient count is observed"
        return self.date_interval[np.argmax(self.patient_counts)]

    def max_interval_patients(self) -> pd.DataFrame:
        "Returns dataframe of the interval with the most patients"
        return self._subset_by_date(self.max_interval())

    def plot_interval_counts(self, title: str) -> None:
        "Plot time patient interval counts"
        plt.plot(self.date_interval, self.patient_counts)
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    # Read in data
    this_dir = Path(__file__).parent.absolute()
    data = pd.read_excel(this_dir / Path("data/XRays.xlsx"))

    # Question 6 & 7
    event_dates = {
        "exam": ["dtBegin", "dtCompleted"],
        "wait": ["dtArrive", "dtBegin"],
    }

    for event, date_range in event_dates.items():
        facility = FacilityTimes(data, *date_range)
        print(
            f"The max patients in a {event} interval is {facility.max_patient_count()} at interval {facility.max_interval()}"
        )
        print(facility.max_interval_patients()[["stMRN"] + date_range])
        facility.plot_interval_counts(
            f"max patients at 30 min intervals ({event} times)"
        )
        print("=" * 60)

    # Question 8
    hospital = FacilityTimes(data, *event_dates["exam"])

    time_ranges = [(7, 9), (10, 12), (13, 15), (14, 16)]
    for time_range in time_ranges:
        print(
            f"At time range {time_range}, there are {hospital.count_patients_at_range(*time_range)} patients"
        )
