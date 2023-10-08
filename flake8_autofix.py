import os
import subprocess as sp
import multiprocessing
import re

omitlist = []


def flake8_file(fpath):
    '''
    see all messages in 1 file
    '''
    file = sp.getoutput('flake8 ' + fpath)
    filelist = file.strip().split('\n')
    filelist = list(filter(None, filelist))
    filelist = [x for x in filelist if x not in omitlist]
    return filelist


def extract_details(entry):
    '''
    fetch information from flake8 message string
    '''
    entrysplit = entry.split(':')
    path = entrysplit[0]
    row = int(entrysplit[1])-1
    col = int(entrysplit[2])-1
    message = ''.join(entrysplit[3])
    return (path, row, col, message, entry)


def get_all_files():
    '''
    get all files that need linting
    '''
    flake8 = sp.getoutput('flake8 ' + os.getcwd())
    flakelist = flake8.strip().split('\n')
    return {x.split(':')[0] for x in flakelist}


def find_fix(message):
    '''
    look through func_fix dict, find key
    '''
    keys = list(func_fix.keys())
    for key in keys:
        if key in message:
            return key
    return None


def delete_line(bundle):
    '''
    delete line of mentioned row
    '''
    print('deleting line')
    lines, details = bundle
    path, row, col, message, entry = details
    with open(path, 'w') as f:
        for index, line in enumerate(lines):
            if index != row:
                f.write(line)


def delete_blank_line(bundle):
    '''
    delete previous line of mentioned row
    '''
    print('deleting blank line')
    lines, details = bundle
    path, row, col, message, entry = details
    with open(path, 'w') as f:
        for index, line in enumerate(lines):
            if index != row-1:
                f.write(line)


def insert_line(bundle):
    '''
    insert an new line at row
    '''
    print('inserting line')
    lines, details = bundle
    path, row, col, message, entry = details
    with open(path, 'w') as f:
        for index, line in enumerate(lines):
            if index == row:
                f.write('\n')
            f.write(line)


def newline_EOF(bundle):
    '''
    insert new line at EOF
    '''
    print('inserting line at end')
    lines, details = bundle
    path, row, col, message, entry = details
    with open(path, 'w') as f:
        for index, line in enumerate(lines):
            f.write(line)
        f.write('\n')


def insert_space_before(bundle):
    '''
    insert space at mentioned col
    '''
    print('inserting space')
    lines, details = bundle
    path, row, col, message, entry = details
    with open(path, 'w') as f:
        for index, line in enumerate(lines):
            if index == row:
                newline = line[:col] + ' ' + line[col:]
                f.write(newline)
            else:
                f.write(line)


def insert_space_after(bundle):
    '''
    insert space after mentioned col
    '''
    print('inserting space')
    lines, details = bundle
    path, row, col, message, entry = details
    with open(path, 'w') as f:
        for index, line in enumerate(lines):
            if index == row:
                newline = line[:col+1] + ' ' + line[col+1:]
                f.write(newline)
            else:
                f.write(line)


def convert_tabs_to_spaces(bundle):
    '''
    convert all tabs to 4 spaces
    '''
    print('converting tabs to spaces')
    lines, details = bundle
    path, row, col, message, entry = details
    with open(path, 'w') as f:
        for index, line in enumerate(lines):
            line = re.sub('\t', '    ', line)
            f.write(line)


def remove_semicolon(bundle):
    '''
    remove semicolons
    '''
    print('deleting semicolon')
    lines, details = bundle
    path, row, col, message, entry = details
    with open(path, 'w') as f:
        for index, line in enumerate(lines):
            if index == row:
                line = line.replace(';', '')
            f.write(line)


def delete_character(bundle):
    '''
    delete a character at row, col
    '''
    print('deleting character')
    lines, details = bundle
    path, row, col, message, entry = details
    with open(path, 'w') as f:
        for index, line in enumerate(lines):
            if index == row:
                line = line[:col] + line[col+1:]
            f.write(line)


def delete_unused_import(bundle):
    '''
    delete a character at row, col
    '''
    print('deleting unused import')
    global omitlist
    lines, details = bundle
    path, row, col, message, entry = details
    string_to_remove = message.split("'")[1].split('.')[-1]
    if string_to_remove in lines[row]:
        with open(path, 'w') as f:
            for index, line in enumerate(lines):
                if index == row:
                    spl = line.split('import')
                    imports = spl[-1].split(',')
                    imports = [x.strip() for x in imports]
                    imports.remove(string_to_remove)
                    if len(imports):
                        new_line = spl[0] + ' import ' + ' ,'.join(imports) + '\n'
                        f.write(new_line)
                else:
                    f.write(line)
    else:
        print("Manually fix", entry)
        omitlist.append(entry)


func_fix = {
    'E201': delete_character,
    'E202': delete_character,
    'E203': delete_character,
    'E211': delete_character,
    'E221': delete_character,
    'E222': delete_character,
    'E225': insert_space_before,
    'E231': insert_space_after,
    'E251': delete_character,
    'E252': insert_space_before,
    'E261': insert_space_before,
    'E262': insert_space_after,
    'E265': insert_space_after,
    'E266': delete_character,
    'E272': delete_character,
    'E302': insert_line,
    'E303': delete_blank_line,
    'E305': insert_line,
    'E703': remove_semicolon,
    'F401': delete_unused_import,
    'W191': convert_tabs_to_spaces,
    'W291': delete_character,
    'W292': newline_EOF,
    'W293': delete_character,
    'W391': delete_line,
}


def solution_selector(full_details):
    '''
    selects the solution to deal out
    '''
    global omitlist
    path, row, col, message, entry = full_details
    # details = (path, row, col, message)
    with open(path, 'r') as f:
        lines = f.readlines()
        key = find_fix(message)
        if key is None:
            print("Manually fix", entry)
            omitlist.append(entry)
        else:
            print(key, entry)
            bundle = (lines, full_details)
            func_fix[key](bundle)


def fix_a_file(file):
    resolved = False
    while not resolved:
        file_errors = flake8_file(file)
        if(len(file_errors) == 0):
            resolved = True
        else:
            details = extract_details(file_errors[0])
            solution_selector(details)


files = get_all_files()
p = multiprocessing.Pool(multiprocessing.cpu_count())
p.map(fix_a_file, files)
