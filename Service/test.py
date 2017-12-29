import time

for i in range(5):
    print(i)
    # time.sleep(5)
L = [x for x in range(5) if x % 2 == 0]
print(L)
print("finished!")