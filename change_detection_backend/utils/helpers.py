import os
import time

def cleanup_old_files(directory, hours=24):
    """Remove files older than specified hours"""
    current_time = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > hours * 3600:  # Convert hours to seconds
                os.remove(file_path)