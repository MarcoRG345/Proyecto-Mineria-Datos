import json
import os

for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.ipynb'):
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if not content.strip():
                        print(f"Empty file found: {path}")
                    else:
                        json.loads(content)
            except Exception as e:
                print(f"Error in {path}: {e}")
