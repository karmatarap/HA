"""
HL7 format
• HL7 transactions are known as messages
• Messages consist of segments, separated with <cr> (new line) – similar to paragraphs in text.
• Each segment starts with its 3‐letter name, followed by its data.
Example: MSH is usually the name of the first segment, “messageheader”
• Segment consists of fields, separated with | (“pipe”)
"""
import re
import datetime
import numpy as np
from typing import List, Union


class HL7:
    """HL7 class contains HL7 text record as a dicti
        properties can be further extended for a full parser. 
    """

    def __init__(self, hl7_text: str) -> None:
        self.hl7_dict = HL7.text_to_dict(hl7_text)

    @staticmethod
    def text_to_dict(hl7_text: str) -> dict:
        """Converts HL7 string to dict
        The key of the dict is the Segment Name
        The values of the dict are the fields.
        This is a naive implementation that does not take into consideration
        segment groups. In the event of segment groups, only the last segment
        in the group is kept.
        """
        return dict(
            (field[0], list(field[1:]))
            for field in (segment.split("|") for segment in hl7_text.split("\n"))
        )

    @property
    def separators(self) -> List[str]:
        "Returns list of separators"
        return list(self.hl7_dict["MSH"][0])

    def _clean(self, text) -> List[str]:
        """Helper function to split texts based on multiple separators
        Separators are removed from the final text"""
        sep_pattern = "|".join(["\\" + x for x in self.separators])
        components = re.split(f"[{sep_pattern}]{{2,}}", text)
        return [re.sub(sep_pattern, " ", c) for c in components]

    @property
    def name(self) -> str:
        "Returns patients short name"
        return self._clean(self.hl7_dict["PID"][4])[0]

    @property
    def sex(self) -> str:
        "Returns Sex stored in field 7 of PID"
        return self.hl7_dict["PID"][7]

    @property
    def dob(self) -> Union[datetime.date, str]:
        "Returns DOB stored in field 6 of PID"
        dob = self.hl7_dict["PID"][6]
        try:
            return datetime.datetime.strptime(dob, "%Y%m%d").date()
        except ValueError:
            return ""


if __name__ == "__main__":
    text = """MSH|^~\&#|NIST^2.16.840.1.113883.3.72.5.20^ISO|NIST^2.16.840.1.113883.3.72.5.21^ISO|NIST^2.16.840.1.113883.3.72.5.22^ISO|NIST^2.16.840.1.113883.3.72.5.23^ISO|20120821140551-0500||ORU^R01^ORU_R01|NIST-ELR-001.01|T|2.5.1|||NE|NE|||||PHLabReport-NoAck^HL7^2.16.840.1.113883.9.11^ISO
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
    patient = HL7(text)
    print(f"Patient name is: {patient.name}")
    print(f"DOB is: {patient.dob}")
    print(f"Sex is: {patient.sex}")
