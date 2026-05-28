import re
import sys

def clean_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Remove all multi-line docstrings (triple quotes)
    # This regex matches """...""" and '''...''' non-greedily.
    # It requires that there's at least one newline inside to be considered "longer than one line".
    content = re.sub(r'\"\"\"[\s\S]*?\n[\s\S]*?\"\"\"', '', content)
    content = re.sub(r"\'\'\'[\s\S]*?\n[\s\S]*?\'\'\'", '', content)
    
    # 2. Remove consecutive # lines
    lines = content.split('\n')
    cleaned_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        if stripped.startswith('#'):
            # Count consecutive comment lines
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith('#'):
                j += 1
            
            consecutive_count = j - i
            if consecutive_count > 1:
                # Skip all these lines
                i = j
                continue
            else:
                # Single comment line, keep it
                cleaned_lines.append(line)
                i += 1
        else:
            cleaned_lines.append(line)
            i += 1
            
    # Remove multiple empty lines
    final_content = '\n'.join(cleaned_lines)
    final_content = re.sub(r'\n{3,}', '\n\n', final_content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_content)

clean_file('camera.py')
print("Cleanup complete.")
