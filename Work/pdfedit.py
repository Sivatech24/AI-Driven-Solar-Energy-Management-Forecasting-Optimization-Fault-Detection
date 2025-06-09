import PyPDF2

def remove_first_page(input_pdf, output_pdf):
    with open(input_pdf, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        writer = PyPDF2.PdfWriter()

        # Add all pages except the first one
        for page_num in range(1, len(reader.pages)):  # Skip first page (index 0)
            writer.add_page(reader.pages[page_num])

        # Save the new PDF without the first page
        with open(output_pdf, "wb") as new_file:
            writer.write(new_file)

# Example usage
remove_first_page("Project Report.pdf", "Project Report Edited.pdf")
