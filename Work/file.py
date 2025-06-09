import json

def convert_inpby_to_py(inpby_path, output_path):
    with open(inpby_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for cell in data.get('cells', []):
            if cell.get('cell_type') == 'code':
                code_lines = cell.get('source', [])
                f_out.write(''.join(code_lines))
                f_out.write('\n\n')  # separate cells by blank lines

# Example usage
convert_inpby_to_py('SolarAnalysis.ipynb', 'SolarAnalysis.py')
