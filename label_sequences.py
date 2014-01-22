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
import matplotlib.pyplot as plt
import numpy as np
import colorsys, os, re, sys

class Application(Frame):

    def change_image(self, ex_ind):
        global ex_file
        if ex_ind >= len(example_ids):
            self.quit()
        try:  # Check if image file already exists
            f = open(get_image_filename(images_folder, example_ids[ex_ind]), 'r')
            f.close()
        except IOError:
            image_from_point_cloud(images_folder, ex_file, ex_ind)
        for idx, val in enumerate([ex_ind - 1, ex_ind]):
            im = Image.open(get_image_filename(images_folder, example_ids[val]))
            im.thumbnail((im_size, im_size), Image.ANTIALIAS)
            imgs[idx] = ImageTk.PhotoImage(im)
            panels[idx].configure(image = imgs[idx])
            panels[idx].image = imgs[idx]

    def next_image(self):
        global ex_index
        ex_index += 1
        self.change_image(ex_index)
        
    def record_yes(self):
        global ex_file, ex_index
        update_predecessor(ex_file, example_ids[ex_index], example_ids[ex_index - 1])
        self.next_image()

    def record_no(self):
        global ex_file, ex_index
        # Indicates this is the start of a sequence
        update_predecessor(ex_file, example_ids[ex_index], example_ids[ex_index])
        self.next_image()

    def undo_record(self):
        global ex_index
        if ex_index == 1:
            return
        ex_index -= 1
        self.change_image(ex_index)

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

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack(side = "bottom")
        self.createWidgets()

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def get_image_filename(images_folder, ex_id):
    return os.path.join(images_folder, "example_%s.png" % (str(ex_id)))

def update_predecessor(h5py_file, ex_id, pred_id):
    if 'pred' in h5py_file[ex_id].keys():
        del h5py_file[ex_id]['pred']
    h5py_file[ex_id]['pred'] = str(pred_id)

def image_from_point_cloud(output_folder, h5py_file, ex_index):
    ex_id = example_ids[ex_index]
    fname = get_image_filename(images_folder, ex_id)
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

if len(sys.argv) < 3:
    print "Usage: ./label_sequences.py <labeled examples file> <output images folder>"
    sys.exit(0)

num_images_shown = 2

labelled_ex_file = sys.argv[1]
ex_file = h5py.File(labelled_ex_file, 'r+')
example_ids = natural_sort(ex_file.keys())
update_predecessor(ex_file, example_ids[0], example_ids[0])

ex_index = 1
images_folder = sys.argv[2]
if images_folder[-1] is not '/':
    images_folder = images_folder + "/"
print "Outputting images to: " , images_folder

root = Tk()
root.wm_title("Label Predecessors")
app = Application(master=root)
app.focus_set()
app.bind('y', lambda event: app.record_yes())
app.bind('n', lambda event: app.record_no())
app.bind('u', lambda event: app.undo_record())

panels = {}
imgs = {}

top_frame = Frame(root)
top_frame.pack()

im_size = (root.winfo_screenwidth()-200) / num_images_shown

for i in [ex_index - 1, ex_index]:
    image_from_point_cloud(images_folder, ex_file, i)
    im = Image.open(get_image_filename(images_folder, example_ids[i]))
    im.thumbnail((im_size, im_size), Image.ANTIALIAS)
    imgs[i] = ImageTk.PhotoImage(im)
    panels[i] = Label(top_frame, image = imgs[i])
    panels[i].pack(side = "left", fill = "both", expand = "yes")

app.mainloop()
ex_file.close()
root.destroy()
