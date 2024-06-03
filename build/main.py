from multiprocessing import Process, freeze_support


def f():
    print("Hello from cx_Freeze")

from 


if __name__ == "__main__":
    freeze_support()
    Process(target=f).start()
