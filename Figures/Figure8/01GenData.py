import os
import shutil

current_dir = os.getcwd()
files_to_copy = ['Data.sh','Data.py','BatchSaveGflops.py', 'NobatchSaveGflops.py']

for item in os.listdir(current_dir):
    item_path = os.path.join(current_dir, item)
    for file in files_to_copy:
        source_file_path = os.path.join(current_dir, file)
        destination_file_path = os.path.join(item_path, file)
        if os.path.isfile(source_file_path):
            try:
                shutil.copy2(source_file_path, destination_file_path)
                print(f"COPY {source_file_path} TO {destination_file_path} SUCCESS")
            except Exception as e:
                print(f"COPY {source_file_path} TO {destination_file_path} FAILED: {e}")

current_dir = os.getcwd()
for item in os.listdir(current_dir):
    item_path = os.path.join(current_dir, item)
    if os.path.isdir(item_path):
        os.chdir(item_path)
        shell_file_path = os.path.join(item_path, 'Data.sh')
        if os.path.isfile(shell_file_path) and os.access(shell_file_path, os.X_OK):
            try:
                os.system(f'./{os.path.basename(shell_file_path)}')
                print(f"Run Data.sh success: {shell_file_path}")
            except Exception as e:
                print(f"Run {shell_file_path} failed: {e}")
        else:
            print(f"File {shell_file_path} not found or not executable.")
        os.chdir(current_dir)