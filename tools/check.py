import json
with open('pulled_notebook.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
notebook = json.loads(data['blob']['source'])
for i, cell in enumerate(notebook.get('cells', [])):
    print(f'Cell {i} type: {cell.get("cell_type")}')
    for output in cell.get('outputs', []):
        print(f'  Output keys: {list(output.keys())}')
        if 'text' in output:
            print('  Length of text:', len(output['text']))
            with open('kaggle_stdout.txt', 'a', encoding='utf-8') as out_f:
                out_f.write(''.join(output['text']))
