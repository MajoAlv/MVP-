import subprocess

subprocess.run(['git', 'add', '.'])
subprocess.run(['git', 'commit', '-m', 'auto: actualización de datos'])
subprocess.run(['git', 'push', 'origin', 'main'])

print("✅ Push exitoso")
