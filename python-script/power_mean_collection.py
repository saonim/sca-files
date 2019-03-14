#!/usr/bin/python
from subprocess import Popen, PIPE
from time import sleep
from sys import argv
import numpy, ivi, os, time, datetime, csv, sys, time

start = time.time()

# setup mso
mso = ivi.agilent.agilentMSOX4104A("TCPIP0::10.75.8.51::INSTR")

trace_num = 10
i = 0
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
file_name = "/home/saoni/exp_results/power_data_"+str(trace_num)+"_"+str(st)+".csv"

# Run 10 times for warm up
for tmp in range(0, 10):
     mso.measurement.initiate()
 #    sleep(0.01)
    
     # Launch the kernel
     os.system('/home/saoni/git-barebone/a.out')
     # sleep
 #    sleep(0.01)

with open(file_name, "wb") as csvfile:

     wr = csv.writer(csvfile, delimiter=',',
                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
     power_mean_list = []

     for tnum in range(0, trace_num):
          mso.measurement.initiate()
          sleep(0.1)
          os.system('/home/saoni/git-barebone/a.out')
          sleep(0.1)
          # read the wave
          wave = mso.channels[0].measurement.fetch_waveform()
          # convert to int
          wave_int = [ord(x) for x in wave]
          # calculate the mean
          cut_off = numpy.mean(wave_int)
          # Noise cancellation ie keep only those less than cut_off 
          filtered_wave = [i for i in wave_int if i < cut_off]
          # Compute mean power
          wave_mean = numpy.mean(filtered_wave)
          if (tnum % 1) == 0:
              print(tnum)
              power_mean_list.append(wave_mean)
              wr.writerow([wave_mean])
              
# mean over all traces
final_mean = numpy.mean(power_mean_list)
print(final_mean)

end = time.time()
print('Time: ', (end - start))

