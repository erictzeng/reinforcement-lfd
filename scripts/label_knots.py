#!/usr/bin/env python

# Run this script with ./label_sequences.py <labeled examples file> <output images folder>
#     e.g. ./label_sequences.py data/misc/labeled_examples.h5 temp
#
# The images in the folder must be named in the following format:
#     example_<id>.png
# where <id> should be the id of the example in the labeled examples file.
#
# This script adds an extra ["pred"] to each of the values of the input h5 file.
# Given the example with id '11', and that f is a handle to the h5 file, then
#     f['11']['pred'] = '10' if example '10' precedes example '11' in knot tying, OR
#     f['11']['pred'] = '11' if example '11' is the start of a knot tying sequence
#
# This script shows the user point clouds of two consecutive labeled examples, and
# asks whether the one on the right follows the one on the left. It iterates through
# the examples in the input file in numerical order (based on id).
#
# The "Undo" button can be used to undo the previous classification.
#
# Shortcuts: 'y'    example on the left is a predecessor of example on the right
#            'n'    example on the left is not a predecessor
#            'u'    undo previous classification
#
# NOTE: If your system does not have ImageTk, then run
#     sudo apt-get install python-imaging-tk

import h5py
from Tkinter import *
from PIL import ImageTk, Image
import IPython as ipy
from pdb import pm, set_trace
import matplotlib.pyplot as plt
import numpy as np
import colorsys, os, re, sys
import argparse

class Application(Frame):

    def __init__(self, infile, outfile, image_folder, im_size, master=None):
        Frame.__init__(self, master)
        self.pack(side = "bottom")
        self.createWidgets()
        self.infile = infile
        self.outfile = outfile
        self.ex_index = 0
        self.num_examples = h5_len(infile)
        self.image_folder = image_folder
        self.im_size = im_size
        self.img = None
        self.panel = None

    def set_panel(self, img, panel):
        self.img = img
        self.panel = panel

    def change_image(self, ex_ind=None):
        if not ex_ind: ex_ind = self.ex_index
        if ex_ind >= self.num_examples:
            self.quit()
        image_from_point_cloud(self.image_folder, self.infile, ex_ind)
        im = Image.open(get_image_filename(self.image_folder, ex_ind))
        im.thumbnail((self.im_size, self.im_size), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(im)
        panel.configure(image = img)
        panel.image = img

    def next_image(self):
        self.ex_index += 1
        self.change_image(self.ex_index)
        
    def record_yes(self):
        set_label(self.outfile, self.infile, self.ex_index, 1)
        self.next_image()

    def record_no(self):
        set_label(self.outfile, self.infile, self.ex_index, 0)
        self.next_image()

    def record_skip(self):
        self.next_image()

    def undo_record(self):
        if self.ex_index == 1:
            return
        self.ex_index -= 1
        del self.outfile[str(len(self.outfile)-1)]
        self.change_image(self.ex_index)

    def createWidgets(self):
        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit
        self.QUIT.pack({"side": "left"})

        self.yes = Button(self)
        self.yes["text"] = "Yes"
        self.yes["command"] = self.record_yes
        self.yes.pack({"side": "left"})

        self.no = Button(self)
        self.no["text"] = "No"
        self.no["command"] = self.record_no
        self.no.pack({"side": "left"})

        self.undo = Button(self)
        self.undo["text"] = "Undo"
        self.undo["command"] = self.undo_record
        self.undo.pack({"side": "left"})

        self.skip = Button(self)
        self.undo["text"] = "Skip"
        self.undo["command"] = self.record_skip
        self.undo.pack({"side": "left"})


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def get_image_filename(images_folder, ex_id):
    return os.path.join(images_folder, "example_%s.png" % (str(ex_id)))

def h5_len(f):
    c = 0
    for i in f.iterkeys():
        c += 1
    return c

def set_label(outfile, infile, ex_id, value):
    new_id = str(h5_len(outfile)) # takes care of adding 1 for us
    g = outfile.create_group(new_id)
    g['knot'] = value
    g['cloud_xyz'] = infile[str(ex_id)]['cloud_xyz'][:]
    
def image_from_point_cloud(output_folder, h5py_file, ex_index):
    ex_id = example_ids[ex_index]
    fname = get_image_filename(output_folder, ex_id)
    cloud_xyz = h5py_file[ex_id]['cloud_xyz']
    plt.clf()
    plt.cla()
    links_z = (cloud_xyz[:-1,2] + cloud_xyz[1:,2])/2.0
    for i in np.argsort(links_z):
        f = float(i)/(links_z.shape[0]-1)
        plt.plot(cloud_xyz[i:i+2,0], cloud_xyz[i:i+2,1], c=colorsys.hsv_to_rgb(f,1,1), linewidth=6)
    plt.axis("equal")
    plt.savefig(fname)
    print "saved ", fname

def flatten_file(f1, tmp_folder):
    from do_task_label import write_flush
    outf = h5py.File(tmp_folder+'flat.h5', 'w')
    for g in f1.itervalues():
        for dset in g.itervalues(): 
            write_flush(outf, [['cloud_xyz', dset[:]]])
    f1.close()
    return outf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('examples_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('image_folder', type=str)    

    args = parser.parse_args()
        
    infile = h5py.File(args.examples_file, 'r')
    example_ids = natural_sort(infile.keys())
    outfile = h5py.File(args.output_file, 'a')

    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    print "Outputting images to: " , args.image_folder

    if 'cloud_xyz' not in iter(infile).next():
        infile = flatten_file(infile, args.image_folder)
            
    root = Tk()
    root.wm_title("Knot or Not")
    im_size = (root.winfo_screenwidth()-200)
    app = Application(infile, outfile, args.image_folder, im_size, master=root)
    app.focus_set()
    app.bind('y', lambda event: app.record_yes())
    app.bind('n', lambda event: app.record_no())
    app.bind('u', lambda event: app.undo_record())
    app.bind('s', lambda event: app.record_skip())
        
    top_frame = Frame(root)
    top_frame.pack()

    image_from_point_cloud(args.image_folder, infile, 0)
    im = Image.open(get_image_filename(args.image_folder, example_ids[0]))
    im.thumbnail((im_size, im_size), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(im)
    panel = Label(top_frame, image = img)
    panel.pack(side = "left", fill = "both", expand = "yes")
    app.set_panel(img, panel)

    try:
        app.mainloop()
    except:        
        outfile.close()
        infile.close()
        root.destroy()
