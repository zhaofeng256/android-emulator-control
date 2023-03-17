import mouse
def callback(e):
    print(e)
mouse.hook(callback)
mouse.wait()