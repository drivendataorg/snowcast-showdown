import time    

def TicTocGenerator():
        ti = 0           # initial time
        tf = time.time() # final time
        while True:
            ti = tf
            tf = time.time()
            yield tf-ti # returns the time difference    
TicToc = TicTocGenerator() # create an instance of the TicTocGen generator