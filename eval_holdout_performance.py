#!/usr/bin/env python

# Run this script with ./eval_holdout_performance.py <images_folder> <output_file>
#     e.g. ./eval_holdout_performance.py data/holdout_set_eval_50 holdout_results.out
#
# The images in the folder must be named in the following format:
#     task_##_step_##.png
#
# This script shows the user images and asks whether they are knots. It iterates
# through the images in chronological order for each knot tying attempt, moving
# on to the next attempt when the user classifies an image as being a knot.
#
# The "Undo No" button can be used to undo an accidental classification of a knot
# as not being a knot.
#
# NOTE: If your system does not have ImageTk, then run
#     sudo apt-get install python-imaging-tk

from Tkinter import *
from PIL import ImageTk, Image
import os, sys

class Application(Frame):
    def get_task(self, index):
        return '_'.join(image_filenames[index].split('_')[0:2])

    def change_image(self, index):
        im = Image.open(images_folder + image_filenames[index])
        img2 = ImageTk.PhotoImage(im)
        panel.configure(image = img2)
        panel.image = img2

    def next_image(self, is_knot):
        global curr_index

        prev_task = ""
        if curr_index > 0:
            prev_task = self.get_task(curr_index - 1)

        if curr_index >= len(image_filenames):
            results[prev_task] = False
            self.quit()
        curr_index += 1

        curr_task = self.get_task(curr_index)
        if curr_task != prev_task and prev_task and prev_task not in results:
            results[prev_task] = False

        if is_knot:
            results[curr_task] = True
            new_task = curr_task

            while new_task == curr_task:
                curr_index += 1
                if curr_index >= len(image_filenames):
                    self.quit()
                new_task = self.get_task(curr_index)

        self.change_image(curr_index)
        
    def record_yes(self):
        self.next_image(True)

    def record_no(self):
        self.next_image(False)

    def undo_record(self):
        global curr_index

        curr_index -= 1
        self.change_image(curr_index)

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
        self.undo["text"] = "Undo No"
        self.undo["command"] = self.undo_record
        self.undo.pack({"side": "left"})

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

images_folder = sys.argv[1]
if images_folder[-1] is not '/':
    images_folder = images_folder + "/"
print "Loading images from: " , images_folder

image_filenames = sorted(os.listdir(images_folder))
curr_index = 0

results = {}
root = Tk()
app = Application(master=root)
img = ImageTk.PhotoImage(Image.open(images_folder + image_filenames[curr_index]))
panel = Label(app, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")
app.mainloop()

accuracy = float(sum(results.values())) / len(results.values())
print "Accuracy: ", accuracy

if len(sys.argv) > 2:
    output_file = sys.argv[2]
    print "Writing results to file ", output_file
    with open (output_file, 'w') as f:
        for k in sorted(results.keys()):
            f.write("%s\t%s\n" % (k, results[k]))
    f.close()

root.destroy()
