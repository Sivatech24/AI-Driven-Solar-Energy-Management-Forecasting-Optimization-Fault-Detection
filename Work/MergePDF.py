import PyPDF2

def merge_pdfs(pdf1, pdf2, output_pdf):
    # Open the PDFs
    with open(pdf1, "rb") as file1, open(pdf2, "rb") as file2:
        reader1 = PyPDF2.PdfReader(file1)
        reader2 = PyPDF2.PdfReader(file2)
        writer = PyPDF2.PdfWriter()

        # Add all pages from both PDFs
        for page in reader1.pages:
            writer.add_page(page)
        for page in reader2.pages:
            writer.add_page(page)

        # Save the merged PDF without compression
        with open(output_pdf, "wb") as merged_file:
            writer.write(merged_file)

# Example Usage
merge_pdfs("Untitled document.pdf", "Project Report Edited.pdf", "merged_output.pdf")
