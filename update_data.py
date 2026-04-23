# %% Correr celda cuando quiere actualizar FT desde compu ABSA
import subprocess
import os

GIT_PATH = r"C:\Users\ftovar\AppData\Local\Programs\Git\cmd\git.exe"
BASE = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE)

subprocess.run([GIT_PATH, 'add', '.'])
subprocess.run([GIT_PATH, 'commit', '-m', 'auto: actualización de datos'])
subprocess.run([GIT_PATH, 'push', 'origin', 'main'])

print("✅ Push exitoso")
# %% Correr celda cuando quiere actualizar FT desde compu personal
import subprocess
import os

GIT_PATH = r"C:\Program Files\Git\cmd\git.exe"
BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

subprocess.run([GIT_PATH, 'pull', 'origin', 'main', '--no-edit'])
subprocess.run([GIT_PATH, 'add', '.'])
subprocess.run([GIT_PATH, 'commit', '-m', 'auto: actualización de datos'])
subprocess.run([GIT_PATH, 'push', 'origin', 'main'])

print("✅ Push exitoso")