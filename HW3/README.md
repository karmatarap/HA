One of the main challenges in any machine learning problem is to discover good features, which may not be readily available in the original data. Sometimes the most primitive data can help discover non-trivial patterns, which can be used in your models. Let’s try?

File XRays contains very basic data: a log of patients coming for their Xray examinations. Imagine a large facility with several Xray units, where a patient can come to have one (or more) Xray exams done. All you have in this log is unique patient IDs (MRN, “Medical Record Number”), and three timestamps: patient arrival time, Xray exam begin and complete (end) time. Can we learn something more interesting from this trivial data?

Hints:
To solve the problems below, you will have to sample facility events at different times of day. For consistency, let’s do this every half hour. That is, you should define your time sampling array as numpy.arange(7, 20, 0.5) – that is, going from 7am to 8pm with 0.5-hour increment. I also suggest plotting each of the metrics below as a function of this time.
Unsurprisingly, the data may contain errors: as we know, timestamps are usually entered manually. Do not make any effort to fix them – take the data as is.

Question 6: (6 pts)
As I mentioned, there are several Xray units (exam rooms) in this facility. At any given time, only at most one patient can be in any Xray room. Let’s also assume, that at some times all rooms were used. How many Xray rooms does this facility have?
4
6
8
10
12
14

Question 7: (6 pts)
The facility realized that at certain times its waiting area gets crowded, and wants to expand it. The waiting area is occupied by the patients only. Based on the data provided, the waiting area should be able to hold:
5 patients
10 patients
15 patients
20 patients
25 patients
30 patients
(note that waiting area renovation is expensive, and the facility does not want to provide more room than needed)

Question 8: (4 pts)
Overall, the peak utilization (total number of Xray exams done during the entire log time) happens during the following time interval:
7:00-9:00
10:00-12:00
13:00-15:00
14:00-16:00
