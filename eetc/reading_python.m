%pyversion('C:\Users\Kevin\Anaconda3\pythonw.exe');
pyversion;  %For me on my Windows machine, I use the Anaconda distribution of Python,
            %so I open an Anaconda prompt, start MATLAB with the command
            %"matlab", and then all my installed Python libraries and modules, even
            %ones from JPL, work.
%Note that MATLAB 2021b works with Python 3.9 and below, but previous versions use
%Python 3.8 or lower.  And the MATLAB website says it doesn't work with
%Windows Store Python.

%If you edit a Python file and save it while this MATLAB file is open, you
%will need to close MATLAB and restart it in order for those changes to
%take effect.

PYpath = fileparts(which('excam_tools.py'));
if count(py.sys.path,PYpath) == 0
    insert(py.sys.path,int32(0),PYpath);
end
flu=1.332525281468406;
flu_b=52.36017227172852;
%flu_bim=52.36017227172852; This is actually only needed for photon
%counting.

%Just work in the folder containing the Python file you want to use.
%eetc module checks if certain inputs are integers, and to enforce integer
%status, I had to use int16(input) for certain inputs below

% Full list of possible inputs (example set of inputs).  You only have to specify target_snr through
% status.  Others after that have default values. 
%                             target_snr=1.1555708791366173e-08,
%                             fluxe=2.06913808111479e-09, 
%                             fluxe_bright=483.2930238571752, 
%                             %fluxe_bim = 484, NOT NEEDED
%                             darke=8.33e-4,  
%                             cic=0.02,
%                             rn=100, 
%                             X=5e4,
%                             a=1.69e-10,
%                             Lij=512,
%                             alpha0=0.75,
%                             fwc=60000,
%                             alpha1=0.75,
%                             fwc_em=100000, 
%                             Nmin=1,
%                             Nmax=49,
%                             tmin=0.264,
%                             tmax=120.,
%                             gmax=5000,
%                             %status=1,  NOT NEEDED
%                             opt_choice=0,
%                             n=4,
%                             Nem=604,
%                             tol=1e-30,
%                             delta_constr=1e-4,
%                             %fault = 250000  NOT NEEDED 
pyOut = py.excam_tools.calc_gain_exptime(12,flu,flu_b,8.33e-4,0.02,40,5e4,1.69e-10, int16(512) ,0.9,int16(50000),0.9,int16(90000),int16(1),int16(49),2,120,5000)
gain = pyOut{1}
frame_time = pyOut{2}
num_frames = pyOut{3}
SNR = pyOut{4}