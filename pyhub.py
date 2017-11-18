import subprocess

process = subprocess.Popen("notepad", stdout=subprocess.PIPE)
output = process.communicate()[0]