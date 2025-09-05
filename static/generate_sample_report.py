from fpdf import FPDF
import os

# Create a PDF object
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# Add content to the PDF
pdf.cell(200, 10, txt="Pneumonia Detection Report", ln=True, align='C')
pdf.ln(10)
pdf.cell(200, 10, txt="Patient Name: John Doe", ln=True)
pdf.cell(200, 10, txt="Diagnosis: Pneumonia Detected", ln=True)
pdf.cell(200, 10, txt="Confidence: 95%", ln=True)

# Save the PDF
output_path = os.path.join("static", "sample_report.pdf")
pdf.output(output_path)

print(f"PDF report saved to {output_path}")
