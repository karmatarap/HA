import signal
import sys
from contextlib import contextmanager
from io import StringIO
from time import sleep, time
from unittest import TestCase, main
import datetime

# import mock
from HW2.hl7_parser import HL7

try:
    # Absent on Windows, trigger AttributeError
    signal.alarm

    def _timeout(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _timeout)

    @contextmanager
    def timeout(seconds=1, message="Timeout!"):
        # NB: doesn't work on windows
        signal.alarm(seconds)
        try:
            yield
        except TimeoutError:
            raise TimeoutError(message)
        finally:
            signal.alarm(0)


except AttributeError:

    @contextmanager
    def timeout(seconds=1, message="Timeout!"):
        t0 = time()
        yield
        if time() - t0 > seconds:
            raise TimeoutError(message)


@contextmanager
def capture_print():
    _stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield sys.stdout
    finally:
        sys.stdout = _stdout


class HW2Tests(TestCase):
    def setUp(self):
        self.text = """MSH|^~\&#|NIST^2.16.840.1.113883.3.72.5.20^ISO|NIST^2.16.840.1.113883.3.72.5.21^ISO|NIST^2.16.840.1.113883.3.72.5.22^ISO|NIST^2.16.840.1.113883.3.72.5.23^ISO|20120821140551-0500||ORU^R01^ORU_R01|NIST-ELR-001.01|T|2.5.1|||NE|NE|||||PHLabReport-NoAck^HL7^2.16.840.1.113883.9.11^ISO
SFT|NIST Lab, Inc.^L^^^^NIST&2.16.840.1.113883.3.987.1&ISO^XX^^^123544|3.6.23|A-1 Lab System|6742873-12||20100617
PID|1||18547545^^^NIST MPI&2.16.840.1.113883.3.72.5.30.2&ISO^MR^University H&2.16.840.1.113883.3.0&ISO~111111111^^^SSN&2.16.840.1.113883.4.1&ISO^SS^SSA&2.16.840.1.113883.3.184&ISO||Lerr^Todd^G.^Jr^^^L~Gwinn^Theodore^F^Jr^^^B|Doolittle^Ramona^G.^Jr^Dr^^M^^^^^^^PhD|20090607|M||2106-3^White^CDCREC^W^White^L^1.1^4|123 North 102nd Street^Apt 4D^Harrisburg^PA^17102^USA^H^^42043~111 South^Apt 14^Harrisburg^PA^17102^USA^C^^42043||^PRN^PH^^1^555^7259890^4^call before 8PM~^NET^Internet^smithb@yahoo.com^^^^^home|^WPN^PH^^1^555^7259890^4^call before 8PM||||||||N^Not Hispanic or Latino^HL70189^NH^Non hispanic^L^2.5.1^4||||||||N|||201206170000-0500|University H^2.16.840.1.113883.3.0^ISO|337915000^Homo sapiens (organism)^SCT^human^human^L^07/31/2012^4
NTE|1|P|Patient is English speaker.|RE^Remark^HL70364^C^Comment^L^2.5.1^V1
NK1|1|Smith^Bea^G.^Jr^Dr^^L^^^^^^^PhD|GRD^Guardian^HL70063^LG^Legal Guardian^L^2.5.1^3|123 North 102nd Street^Apt 4D^Harrisburg^PA^17102^USA^H^^42043|^PRN^PH^^1^555^7259890^4^call before 8PM~^NET^Internet^smithb@yahoo.com^^^^^home
PV1|1|O||C||||||||||||||||||||||||||||||||||||||||20120615|20120615
ORC|RE|TEST000123A^NIST_Placer _App^2.16.840.1.113883.3.72.5.24^ISO|system generated^NIST_Sending_App^2.16.840.1.113883.3.72.5.24^ISO|system generated^NIST_Sending_App^2.16.840.1.113883.3.72.5.24^ISO||||||||111111111^Bloodraw^Leonard^T^JR^DR^^^NPI&2.16.840.1.113883.4.6&ISO^L^^^NPI^NPI_Facility&2.16.840.1.113883.3.72.5.26&ISO^^^^^^^MD||^WPN^PH^^1^555^7771234^11^Hospital Line~^WPN^PH^^1^555^2271234^4^Office Phone|||||||University Hospital^L^^^^NIST sending app&2.16.840.1.113883.3.72.5.21&ISO^XX^^^111|Firstcare Way^Building 1^Harrisburg^PA^17111^USA^L^^42043|^WPN^PH^^1^555^7771234^11^Call  9AM  to 5PM|Firstcare Way^Building 1^Harrisburg^PA^17111^USA^B^^42043
OBR|1|TEST000123A^NIST_Placer _App^2.16.840.1.113883.3.72.5.24^ISO|system generated^NIST_Sending_App^2.16.840.1.113883.3.72.5.24^ISO|5671-3^Lead [Mass/volume] in Blood^LN^PB^lead blood^L^2.40^1.2|||20120615|20120615|||||Lead exposure|||111111111^Bloodraw^Leonard^T^JR^DR^^^NPI&2.16.840.1.113883.4.6&ISO^L^^^NPI^NPI_Facility&2.16.840.1.113883.3.72.5.26&ISO^^^^^^^MD|^WPN^PH^^1^555^7771234^11^Hospital Line~^WPN^PH^^1^555^2271234^4^Office Phone|||||201206170000-0500|||F||||||V1586^HX-contact/exposure lead^I9CDX^LEAD^Lead exposure^L^29^V1|111&Varma&Raja&Rami&JR&DR&PHD&&NIST_Sending_App&2.16.840.1.113883.3.72.5.21&ISO
OBX|1|SN|5671-3^Lead [Mass/volume] in Blood^LN^PB^lead blood^L^2.40^V1||=^9.2|ug/dL^microgram per deciliter^UCUM^ug/dl^microgram per deciliter^L^1.1^V1|0.0 - 5.0|H^Above High Normal^HL70078^H^High^L^2.7^V1|||F|||20120615|||0263^Atomic Absorption Spectrophotometry^OBSMETHOD^ETAAS^Electrothermal Atomic Absorption Spectrophotometry^L^20090501^V1||20120617||||University Hospital Chem Lab^L^^^^CLIA&2.16.840.1.113883.4.7&ISO^XX^^^01D1111111|Firstcare Way^Building 2^Harrisburg^PA^17111^USA^L^^42043|1790019875^House^Gregory^F^III^Dr^^^NPI&2.16.840.1.113883.4.6&ISO^L^^^NPI^NPI_Facility&2.16.840.1.113883.3.72.5.26&ISO^^^^^^^MD
SPM|1|^SP004X10987&Filler_LIS&2.16.840.1.113883.3.72.5.21&ISO||440500007^Capillary Blood Specimen^SCT^CAPF^Capillary, filter paper card^L^07/31/2012^v1|73775008^Morning (qualifier value)^SCT^AM^A.M. sample^L^07/31/2012^40939|NONE^none^HL70371^NA^No Additive^L^2.5.1^V1|1048003^Capillary Specimen Collection (procedure)^SCT^CAPF^Capillary, filter paper card^L^07/31/2012^V1|7569003^Finger structure (body structure)^SCT^FIL^Finger, Left^L^07/31/2012^V1|7771000^Left (qualifier value)^SCT^FIL^Finger, Left^L^07/31/2012^V1||P^Patient^HL70369^P^Patient^L^2.5.1^V1|1^{#}&Number&UCUM&unit&unit&L&1.1&V1|||||20120615^20120615|20120617100038
OBX|1|SN|35659-2^Age at Specimen Collection^LN^AGE^AGE^L^2.40^V1||=^3|a^Year^UCUM^Y^Years^L^1.1^V1|||||F|||20120615|||||20120617||||University Hospital Chem Lab^L^^^^CLIA&2.16.840.1.113883.4.7&ISO^XX^^^01D1111111|Firstcare Way^Building 2^Harrisburg^PA^17111^USA^L^^42043|1790019875^House^Gregory^F^III^Dr^^^NPI&2.16.840.1.113883.4.6&ISO^L^^^NPI^NPI_Facility&2.16.840.1.113883.3.72.5.26&ISO^^^^^^^MD"""

        self.patient = HL7(self.text)

    def test_segment_ids(self):
        self.assertEqual(
            list(self.patient.hl7_dict.keys()),
            [
                "MSH",
                "SFT",
                "PID",
                "NTE",
                "NK1",
                "PV1",
                "ORC",
                "OBR",
                "OBX",
                "SPM",
            ],
        )

    def test_sex(self):
        self.assertEqual(self.patient.sex, "M")

    def test_dob(self):
        self.assertEqual(self.patient.dob, datetime.date(2009, 6, 7))

    def test_name(self):
        self.assertEqual(self.patient.name, "Lerr Todd G. Jr")


if __name__ == "__main__":
    main()
