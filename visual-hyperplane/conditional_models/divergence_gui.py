import sys
import json
import os
import matplotlib
import Tkinter as tk
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import style
style.use('ggplot')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


from platform import system as platform

# set up your Tk Frame and whatnot here...


root_dir = 'runs/'

# Get all subfolders
subfolders = list(sorted([o for o in os.listdir(root_dir)
              if os.path.isdir(os.path.join(root_dir,o))]))
print 'subfolders', subfolders

# For each subfolder get arguments
args = []
for subfolder in subfolders:
    path = os.path.join(root_dir, subfolder)
    with open('{}/args.json'.format(path), 'rb') as fp:
        arg = json.load(fp)
        args.append(arg)

# For each subfolder get stats
stats = []
all_stats = set()
for subfolder in subfolders:
    path = os.path.join(root_dir, subfolder)
    with open('{}/stats.json'.format(path), 'rb') as fp:
        stat = json.load(fp)
        stats.append(stat)
        all_stats.update(stat)
all_stats = list(sorted(all_stats))
print 'All stats', all_stats


from Tkinter import *

master = Tk()
master.geometry('800x600')

selectFrame = Frame(master)
selectFrame.pack( side = LEFT )

from platform import system as platform

# set up your Tk Frame and whatnot here...

if platform() == 'Darwin':  # How Mac OS X is identified by Python
    os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

# Current run, for which info should be displayed
focused_run = None

def run_update(evt):
    global focused_run
    print 'run update'
    w = evt.widget
    index = int(w.curselection()[0])
    value = w.get(index)
    print 'You selected item %d: "%s"' % (index, value)
    focused_run = index

    update()

def stat_update(evt):
    w = evt.widget
    index = int(w.curselection()[0])
    value = w.get(index)
    print 'You selected item %d: "%s"' % (index, value)

    update()

def update():
    # Note here that Tkinter passes an event object to onselect()

    print run_listbox.curselection()
    print stat_listbox.curselection()


    # Print current arguments
    print 'focused_run', focused_run
    if focused_run is not None:
        print 'updating text'
        args_text = []
        print args[focused_run]
        for key, val in args[focused_run].items():
            args_text.append('{}: {}'.format(key,val))
        args_text = u'\n'.join(args_text)
        argsBox.configure(text=args_text)

    # Smooth if needed
    smooth_window = smoothingScale.get()
    smooth_window = np.clip(1+int((smooth_window-1)/2)*2, 1, 101)
    print 'Smooth window', smooth_window

    # Update plot if plot is selected
    if stat_listbox.curselection() and run_listbox.curselection():
        ax.cla()
        stat_idx = stat_listbox.curselection()[0]
        stat_name = stat_listbox.get(stat_idx)
        print 'Plotting', stat_name
        for run in run_listbox.curselection():
            stat = stats[run][stat_name]
            # Smooth the stat
            smoothed = gaussian_filter1d(stat, smooth_window)

            ax.plot(smoothed, label=subfolders[run])
            ax.set_xlabel('iterations')
            ax.set_title(stat_name)
        ax.legend()
        canvas.draw()


w = Label(selectFrame, text="List of runs:")
w.pack()

run_listbox = Listbox(selectFrame, selectmode=EXTENDED, exportselection=0)
run_listbox.pack(side=TOP)
for subfolder in subfolders:
    run_listbox.insert(END, subfolder)
run_listbox.bind('<<ListboxSelect>>', run_update)

w = Label(selectFrame, text="Stats:")
w.pack()

stat_listbox = Listbox(selectFrame, selectmode=BROWSE, exportselection=0)
stat_listbox.pack(side=TOP)
for stat in all_stats:
    stat_listbox.insert(END, stat)
stat_listbox.bind('<<ListboxSelect>>', stat_update)

argsBox = Message(selectFrame, text="details", width=300)
argsBox.pack(side=BOTTOM)

# Canvas frame
plotFrame = Frame(master)
plotFrame.pack(side=LEFT)

f = Figure(figsize=(5, 5), dpi=100)
ax = f.add_subplot(111)
ax.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])

canvas = FigureCanvasTkAgg(f, plotFrame)
canvas.draw()
canvas.get_tk_widget().pack(side=TOP, fill=tk.BOTH, expand=True)

toolbar = NavigationToolbar2TkAgg(canvas, plotFrame)
toolbar.update()
canvas._tkcanvas.pack(side=TOP, fill=tk.BOTH, expand=True)

# Smoothing
w = Label(plotFrame, text="Smoothing window size:")
w.pack(side=TOP)
smoothingScale = Scale(plotFrame, from_=1, to=101, resolution=1, orient=HORIZONTAL)
smoothingScale.pack(side=TOP)
smoothingScale.set(21)
smoothingScale.bind("<ButtonRelease-1>", lambda x:update())

mainloop()


