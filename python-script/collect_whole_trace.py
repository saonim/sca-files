#!/usr/bin/python
from subprocess import Popen, PIPE
from time import sleep
from sys import argv
import numpy, ivi, os, time, datetime, csv, sys, time

start = time.time()

# setup mso
mso = ivi.agilent.agilentMSOX4104A("TCPIP0::129.10.61.226::INSTR")

trace_num = 10000
triggered_trace_num = 0
i = 0
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
file_name = "/home/saoni/exp_results/attack_data_"+str(trace_num)+"_"+str(st)+".csv"

# Run 10 times for warm up
for tmp in range(0, 10):
    mso.measurement.initiate()
    sleep(0.01)
    
    # Launch the kernel
    os.system('/home/saoni/empty_kernel/a.out')
    # sleep
    sleep(0.01)

with open(file_name, "wb") as csvfile:

     wr = csv.writer(csvfile, delimiter=',',
                     quotechar='|', quoting=csv.QUOTE_MINIMAL)

     #for tnum in range(0, trace_num):
     while (triggered_trace_num < trace_num):
          mso.measurement.initiate()
          sleep(0.1)
          os.system('/home/saoni/empty_kernel/a.out')
          sleep(0.1)
          # read the wave
          wave = mso.channels[0].measurement.fetch_waveform()
          # convert to int
          wave_int = [ord(x) for x in wave]
          # calculate the mean
          cut_off = numpy.mean(wave_int)
          #print(cut_off)
          # if > 150, it is not triggered
          if (cut_off < 150):
              #print("triggered")
              triggered_trace_num = triggered_trace_num + 1
              # Noise cancellation ie keep only those less than cut_off 
              filtered_wave = [i for i in wave_int if i < cut_off]
              # Write the filtered wave to a csv          
              wr.writerow([filtered_wave])
              
end = time.time()
print('Time: ', (end - start))

