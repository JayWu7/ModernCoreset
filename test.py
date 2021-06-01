import time

a = 10000000000 

start = time.time()
for _ in range(a):
    100 * 10.2
end = time.time()

print(end-start)


