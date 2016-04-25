import subprocess

a = subprocess.Popen(["python", "wwww.py","&"],stdin=subprocess.PIPE)
for i in range(1,100):
    print a.communicate(input=(str(i)+'\n'))
    a.wait()

