import os
import glob

def get_list_path(path):
    exts = ['*.txt']
    list_path = []
    for ext in exts:
        list_ext = glob.glob(os.path.join(path, '**', ext), recursive=True)
        list_path.extend(list_ext)
    return list_path

def get_chars(path, output='chars.txt'):
    dict_chars = {}
    list_path = get_list_path(path)
    for path_txt in list_path:
        with open(path_txt, 'r', encoding='utf-8') as f:
            text = f.read()
            for char in text:
                # Ensure the character is printable, not whitespace, and not already in dict
                if char.isprintable() and not char.isspace() and char not in dict_chars:
                    dict_chars[char] = None
    
    with open(output, 'w', encoding='utf-8') as f:
        f.write(''.join(dict_chars.keys()))
    
    print('DONE!')

if __name__ == '__main__':
    path = '.'
    output='chars.txt'
    get_chars(path=path, output=output)
