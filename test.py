from pypdf import Pdfreader 

reader = Pdfreader("sample_medical_insurance.pdf")
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()