# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits import mplot3d

from ..excam_tools import calc_gain_fixed_Ntime
import excam_tools as excam
import excam_tools_agreement as agree
import excam_tools_full as full
import excam_tools_res_retry as res

#flu = np.logspace(-9, 3, num=39, base=10);  #np.arange(0,10,0.1);
flu = np.logspace(-5, 3, num=27, base=10);
#flu_b = np.logspace(-9, 3 ,num=39, base=10); #np.arange(0,10,0.1);
flu_b = np.logspace(-5, 3 ,num=27, base=10);
XX,YY = np.meshgrid(flu,flu_b);
exp_time = np.zeros([len(flu),len(flu_b)]);
opt_2 =np.zeros([len(flu),len(flu_b)]);
color = np.full([len(flu),len(flu_b)],'blue'); #color to show which points were 2nd optimization
opt_1 = np.zeros([len(flu),len(flu_b)]);
opt_no = np.zeros([len(flu),len(flu_b)]);
opt_3 = np.zeros([len(flu),len(flu_b)]);
opt_4 = np.zeros([len(flu),len(flu_b)]);
snr = np.zeros([len(flu),len(flu_b)]);
runtime = np.zeros([len(flu),len(flu_b)]);
opt_9 = 0;
no_success_1_break = 0;
no_success_gmax_break = 0;
success_1_break = 0;
success_gmax_break = 0;
none_of_the_above = 0;
opt_status = np.zeros([len(flu),len(flu_b)]);
opt_11 = np.zeros([len(flu),len(flu_b)]);
opt_12 = np.zeros([len(flu),len(flu_b)]);
opt_13 = np.zeros([len(flu),len(flu_b)]);
opt_14 = np.zeros([len(flu),len(flu_b)]);
opt_15 = np.zeros([len(flu),len(flu_b)]);
opt_16 = np.zeros([len(flu),len(flu_b)]);
opt_17 = np.zeros([len(flu),len(flu_b)]);
opt_18 = np.zeros([len(flu),len(flu_b)]);
opt_19 = np.zeros([len(flu),len(flu_b)]);
opt_20 = np.zeros([len(flu),len(flu_b)]);

opt_21 = np.zeros([len(flu),len(flu_b)]);
opt_22 = np.zeros([len(flu),len(flu_b)]);
opt_23 = np.zeros([len(flu),len(flu_b)]);
opt_24 = np.zeros([len(flu),len(flu_b)]);
opt_25 = np.zeros([len(flu),len(flu_b)]);
opt_26 = np.zeros([len(flu),len(flu_b)]);
opt_27 = np.zeros([len(flu),len(flu_b)]);

delta_constr=1e-4;
times = 0;
pc_0 = 0;
pc_1 = 0;
timeratio_sum = 0;
timeratio_min = 1;
denom = 0;
pctime_counter=0;
pctime_tot = 0;
regtime_tot = 0;
maxN = 0;
counter=0;
for xi, x in np.ndenumerate(flu):
    for yi, y in np.ndenumerate(flu_b):
        if y > x:
            counter= counter +1;
            opt=10; #something besides 0 and 1; if algorithm fails completely, opt will remain 2
            start_time = time.time()

            [gain,t,Nf,SNR, opt] = excam.calc_pc(#target_snr=args.snr,
                            target_snr=1e-8,  #1e-8,  #achievable for any part of parameter space
                            fluxe=x,
                            fluxe_bright = y,
                            darke=8.33e-4,
                            cic=0.02,
                            rn=100,  #about 40e- typically?
                            X=5e4,
                            a=1.69e-10,
                            Lij=512,
                            alpha0=0.75,
                            fwc=60000,
                            alpha1=0.75,
                            fwc_em=100000,
                            Nmin=1,
                            Nmax=25200,
                            tmin=.001,
                            tmax=6300,
                            #t = 10,
                            gmax=5000,
                            pc_ecount_max = 0.1,
                            fault = 250000,
                            fluxe_bim = None,
                            #status=1,
                            opt_choice=0,
                            n=4,
                            Nem=604,
                            delta_constr=delta_constr
                            )
            runtime[yi,xi] = time.time()-start_time
            #print(gain, t, Nf, SNR, opt);
            if opt==0 or opt==1:
                snr[yi,xi] = SNR;
                exp_time[yi,xi] = t*Nf;
            if opt == 0:
                opt_1[yi,xi] = 1;
            if opt == 1:
                opt_2[yi,xi] = 1;
                color[yi,xi] = 'red';
            #if opt == 2:
            #    opt_no[yi,xi] = 1;
            if opt == 3:
                opt_3[yi,xi] = 1;
            #if opt == 4:
            #    none_of_the_above += 1;
            # if opt == 5:
            #     no_success_1_break = no_success_1_break + 1;
            # if opt == 6:
            #     no_success_gmax_break +=1;
            # if opt == 7:
            #     success_1_break +=1;
            # if opt == 8:
            #     success_gmax_break +=1;
            if opt == 9:
                opt_9 += 1;

            #res.status == 0:
            if opt == 11:
                opt_status[yi,xi] = 1; color[yi,xi] = 'blue'; opt_11[yi,xi]=1;
            #res.status == 1:
            if opt == 12:
                opt_status[yi,xi] = 1; color[yi,xi] = 'red'; opt_12[yi,xi]=1;
            #res.status == 2:
            if opt == 13:
                opt_status[yi,xi] = 1; color[yi,xi] = 'c'; opt_13[yi,xi]=1;
            #res.status == 3:
            if opt == 14:
                opt_status[yi,xi] = 1; color[yi,xi] = 'g'; opt_14[yi,xi]=1;
            #res.status == 4: (inequality constraints incompatible)
            if opt == 15:
                opt_status[yi,xi] = 1; color[yi,xi] = 'aquamarine'; opt_15[yi,xi]=1;
            #res.status == 5:
            if opt == 16:
                opt_status[yi,xi] = 1; color[yi,xi] = 'm'; opt_16[yi,xi]=1;
            #res.status == 6:
            if opt == 17:
                opt_status[yi,xi] = 1; color[yi,xi] = 'y'; opt_17[yi,xi]=1;
            #res.status == 7:
            if opt == 18:
                opt_status[yi,xi] = 1; color[yi,xi] = 'k'; opt_18[yi,xi]=1;
            #res.status == 8: (positive directional derivative in linsearch)
            if opt == 19:
                opt_status[yi,xi] = 1; color[yi,xi] = 'pink'; opt_19[yi,xi]=1;
            #res.status == 9: (iteration limit reached)
            if opt == 20:
                opt_status[yi,xi] = 1; color[yi,xi] = 'tan'; opt_20[yi,xi]=1;

            #new test
            if opt == 21:
                opt_21[yi,xi] = 1;
            if opt == 22:
                opt_22[yi,xi] = 1;
            if opt == 23:
                opt_23[yi,xi] = 1;
            if opt == 24:
                opt_24[yi,xi] = 1;
            if opt == 25:
                opt_25[yi,xi] = 1;
            if opt == 26:
                opt_26[yi,xi] = 1;
            if opt == 27:
                opt_27[yi,xi] = 1;

            if True:#opt ==1 or opt == 2:
                rerun= excam.calc_pc(#target_snr=args.snr,
                            target_snr=5,#SNR, #/(1+delta_constr) used to cancel lower bound in snr constraint, #float(f'{SNR:.7g}'),
                            fluxe=x,
                            fluxe_bright = y,
                            darke=8.33e-4,
                            cic=0.04,
                            rn=120,  #about 40e- typically?
                            X=5e4,
                            a=1.69e-10,
                            Lij=512,
                            alpha0=0.75,
                            fwc=50000,
                            alpha1=0.75,
                            fwc_em=50000,
                            Nmin=1,
                            Nmax=25200,
                            tmin=1,
                            tmax=6300,
                            #t = 10,
                            gmax=5000,
                            pc_ecount_max = 0.1,
                            fault = 250000,
                            fluxe_bim = None,
                            #status=1,
                            opt_choice=1,
                            n=4,
                            Nem=604,
                            delta_constr=delta_constr
                            )
                rerun2 = excam.calc_gain_exptime(
                    target_snr=5,#SNR, #/(1+delta_constr) used to cancel lower bound in snr constraint, #float(f'{SNR:.7g}'),
                            fluxe=x,
                            fluxe_bright = y,
                            darke=8.33e-4,
                            cic=0.04,
                            rn=120,  #about 40e- typically?
                            X=5e4,
                            a=1.69e-10,
                            Lij=512,
                            alpha0=0.75,
                            fwc=50000,
                            alpha1=0.75,
                            fwc_em=50000,
                            Nmin=1,
                            Nmax=25200,
                            tmin=1,
                            tmax=6300,
                            #t = 10,
                            gmax=5000,
                            pc_ecount_max = 0.1,
                            fault = 250000,
                            fluxe_bim = None,
                            #status=1,
                            opt_choice=1,
                            n=4,
                            Nem=604,
                            delta_constr=delta_constr
                            )
                if rerun[-1] == 1:
                    pc_1 = pc_1 + 1;
                if rerun[-1] == 0:
                    pc_0 = pc_0 + 1;
                    if rerun[2] > maxN:
                        maxN = rerun[2]
                if rerun[-1] == 1 and rerun2[-1]==1 and x < 0.1/1: #pc_ecount_max/tmin
                    timeratio_sum += rerun[2]*rerun[1]/(rerun2[2]*rerun2[1])
                    pctime_tot += rerun[2]*rerun[1]
                    regtime_tot += rerun2[2]*rerun2[1]
                    denom += 1
                    if timeratio_min > rerun[2]*rerun[1]/(rerun2[2]*rerun2[1]):
                        timeratio_min = rerun[2]*rerun[1]/(rerun2[2]*rerun2[1])
                    if (1-delta_constr)*0.1/x > rerun[1]:
                        pctime_counter+=1;
                if rerun[-1] == 1 and rerun2[-1]==1 and rerun2[-2]/rerun[-2] > 1+delta_constr and rerun2[2]*rerun2[1]/(rerun[2]*rerun[1]) < 1-delta_constr:
                    times= times+1;
                    print("pc snr: ", rerun[-2], " , analog snr bigger: ", rerun2[-2], ", fluxe: ", x, ", times: ", times)
                if rerun[-1]!=0:
                    opt_4[yi,xi]=1;
                    #print("fluxe= ", x, " fluxe_bright= ", y, " target_snr= ",SNR)

opt_1_total=np.sum(opt_1);
opt_2_total=np.sum(opt_2);
print("delta_constr: ", delta_constr)
print("number of runs that were viable for photon counting: ", pc_1)
print("max N for target snr = 5: ", maxN)
print('average ratio of time_pc/time_analog: ', (pctime_tot/denom)/(regtime_tot/denom), ", # of runs: ", denom )#timeratio_sum/denom)
print('minimum ratio of time_pc/time_analog: ', timeratio_min)
print("number of times pc time wasn't maximal: ", pctime_counter)
print("number of times opt=0:  ",opt_1_total, '\n')
#print("number of times 2nd optimization worked:  ",opt_2_total, '\n')
print("number of times opt=1:  ",opt_2_total, '\n')
print("number of times integration time or snr values didn't agree (opt=9; only relevant when running agree file): ",opt_9, '\n')
#print("number of times both g=1 and gmax worked (opt=9):  ", opt_9, '\n')
#print("number of times both optimizations failed (opt=2):  ",np.sum(opt_no), '\n')
print("number of times opt=3:  ",np.sum(opt_3), '\n')
#print("number of times snr disagreement was more than half an order of magnitude off (opt=3):  ", np.sum(opt_3), '\n')
print("number of times 1st optimization can't reach the max snr as its target:  ", np.sum(opt_4), '\n')
#print("number of times both optimizations failed but outputs within 1/2 order of mag, but result broke constraints:  ", constraint_breaks, '\n')
#print("number of g=1 unsuccessful runs that broke constraints:  ", no_success_1_break, '\n')
#print("number of g=gmax unsuccessful runs that broke constraints:  ", no_success_gmax_break, '\n')
#print("number of g=1 successful runs that broke constraints:  ", success_1_break, '\n')
#print("number of g=gmax successful runs that broke constraints:  ", success_gmax_break, '\n')
#print("number of times runs fell into none of the above categories:  ",none_of_the_above, '\n')
print("total number of runs:  ",counter)
print("average SNR:  ",np.sum(snr)/(opt_1_total+opt_2_total),'\n')
print("average N*tfr:  ",np.sum(exp_time)/(opt_1_total+opt_2_total),'\n')
print("average run time: ",np.sum(runtime)/counter)

print("opt_21: ",np.sum(opt_21))
print("opt_22: ",np.sum(opt_22))
print("opt_23: ",np.sum(opt_23))
print("opt_24: ",np.sum(opt_24))
print("opt_25: ",np.sum(opt_25))
print("opt_26: ",np.sum(opt_26))
print("opt_27: ",np.sum(opt_27))

def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"  # remove int() if you don't use MaxNLocator
    # return f"{10**val:.2e}"      # e-Notation

# opt_status[opt_status==0] = np.nan
# print(flu)
# print(flu_b)
# print(opt_status)
# print(opt_status[opt_status==1].size)
# plt.figure();
# axes1 = plt.axes(projection='3d')
# axes1.plot_surface(XX,YY, opt_status, facecolors = color)
# axes1.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
# axes1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
# axes1.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
# axes1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
# plt.xlabel('fluxe (electrons/s)'); plt.ylabel('fluxe_bright (electrons/s)')
# plt.title('res.status');
# plt.show()

# entries wouldn't be 0 for physical snr or exp_time
exp_time[exp_time==0] = np.nan  #for plotting purposes; also, any entries for which
#optimization didn't work at all will not show up and would look like holes in the plot
snr[snr==0] = np.nan  #for plotting purposes

plt.figure();
#fig, axes = plt.subplots(subplot_kw=dict(projection='3d'))
axes = plt.axes(projection='3d')
#axes.plot_surface(XX,YY,exp_time, rstride=1, cstride=1, edgecolors='k', lw=0.6, facecolors = color)
axes.plot_surface(XX,YY,exp_time, facecolors = color)
plt.xlabel('fluxe (electrons/s)'); plt.ylabel('fluxe_bright (electrons/s)')
plt.title('N*tfr for degraded detector optimization');
plt.show()

plt.figure();
axes1 = plt.axes(projection='3d')
axes1.plot_surface(XX,YY,snr, facecolors = color)
plt.xlabel('fluxe (electrons/s)'); plt.ylabel('fluxe_bright (electrons/s)')
plt.title('SNR for degraded detector optimization');
plt.show()