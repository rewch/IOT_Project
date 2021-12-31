import time
import threading
from threading import Timer
from concurrent.futures import ThreadPoolExecutor, as_completed

# def func(a, b):
#     print("Called function")
#     return a * b
 
# # Schedule a timer for 5 seconds
# # We pass arguments 3 and 4
# t = Timer(5.0, func, [3, 4])

# start_time = time.time()
 
# # Start the timer
# t.start()
 
# end_time = time.time()
 
# if end_time - start_time < 5.0:
#     print("Timer will wait for sometime before calling the function")
# else:
#     print("5 seconds already passed. Timer finished calling func()")

 
# Global Variable for the Thread ID Number
# tid = 0
# # Spawn 4 threads
# NUM_THREADS = 4
 
# def func(arg):
#     time.sleep(1)
#     return arg * arg
 
# if __name__ == '__main__':
#     with ThreadPoolExecutor() as thread_executor:
#         start_time = time.time()
#         # Going to spawn threads
#         tid += NUM_THREADS
#         # Await all results
#         await_results = [thread_executor.submit(func, arg=i) for i in range(1, tid+1)]
#         for f in as_completed([future for future in await_results]):
#             print(f.result())
#         end_time = time.time()
#         print(f"Total Time taken for {NUM_THREADS} threads: {end_time - start_time}")

# import threading
# def set_interval(func, sec):
#     def func_wrapper():
#         set_interval(func, sec) 
#         func()  
#     t = threading.Timer(sec, func_wrapper)
#     t.start()
#     return 
    



# StartTime=time.time()

# def action() :
#     print('action ! -> time : {:.1f}s'.format(time.time()-StartTime))


# class setInterval :
#     def __init__(self,interval,action) :
#         self.interval=interval
#         self.action=action
#         self.stopEvent=threading.Event()
#         thread=threading.Thread(target=self.__setInterval)
#         thread.start()

#     def __setInterval(self) :
#         nextTime=time.time()+self.interval
#         while not self.stopEvent.wait(nextTime-time.time()) :
#             nextTime+=self.interval
#             self.action()

#     def cancel(self) :
#         self.stopEvent.set()

# # start action every 0.6s
# inter=setInterval(0.6,action)
# print('just after setInterval -> time : {:.1f}s'.format(time.time()-StartTime))

# # will stop interval in 5s
# # t=threading.Timer(5,inter.cancel)
# inter.cancel()



# def x():
#     while True :
#         print("Hello")
#         time.sleep(1)

# thread1 = threading.Thread(target=x)
# thread1.start()
# print("World")
msg = "absd"

print(msg[1:])