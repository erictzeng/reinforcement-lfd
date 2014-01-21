#!/usr/bin/env python

# Run this script with ./eval_holdout_performance.py images_folder failure file <output_file>
#     e.g. ./eval_holdout_performance.py data/results/multi_quad_100/images data/results/multi_quad_100/holdout_failures.out data/results/multi_quad_100/holdout_results.out
#
# The images in the folder must be named in the following format:
#     task_##_step_##.png
#
# It is assumed that all the knot tying attempts (i.e task_##) have the same number
# of steps.
#
# This script shows the user images and asks whether they are knots. It iterates
# through the images in chronological order for each knot tying attempt, moving
# on to the next attempt when the user classifies an image as being a knot.
#
# The "Undo" button can be used to undo the previous classification of a task.
#
# Shortcuts: 'y'    is a knot
#            'n'    is not a knot
#            'u'    undo classification of past image as not being a knot
#            'f'    record an interesting failure into output file
#
# NOTE: If your system does not have ImageTk, then run
#     sudo apt-get install python-imaging-tk

from Tkinter import *
from PIL import ImageTk, Image
import IPython as ipy
import os, sys

def get_task(index, t_index = True):
    if t_index:
        index = index*num_steps
    return '_'.join(image_filenames[index].split('_')[0:2])

class Application(Frame):

    def change_image(self, t_index):
        start_index = t_index * num_steps
        if start_index >= len(image_filenames):
            self.quit()
        for i in range(num_steps):
            im = Image.open(images_folder + image_filenames[start_index + i])
            im.thumbnail((im_size, im_size), Image.ANTIALIAS)
            imgs[i] = ImageTk.PhotoImage(im)
            panels[i].configure(image = imgs[i])
            panels[i].image = imgs[i]

    def next_image(self):
        global task_index
        task_index += 1
        self.change_image(task_index)
        
    def record_yes(self):
        results[get_task(task_index)] = True
        self.next_image()

    def record_no(self):
        results[get_task(task_index)] = False
        self.next_image()

    def undo_record(self):
        global task_index
        task_index -= 1
        task_name = get_task(task_index)
        if task_name in failures:
            failures.remove(task_name)
        if task_name in results:
            results.pop(task_name)
        self.change_image(task_index)

    def note_failure(self):
        failures.append(get_task(task_index))
        self.record_no()

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

        self.failure = Button(self)
        self.failure["text"] = "Note Failure"
        self.failure["command"] = self.note_failure
        self.failure.pack({"side": "left"})

        self.undo = Button(self)
        self.undo["text"] = "Undo"
        self.undo["command"] = self.undo_record
        self.undo.pack({"side": "left"})

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack(side = "bottom")
        self.createWidgets()

if len(sys.argv) < 3:
    print "Usage: ./eval_holdout_performance.py <images_folder> <failure file> [output_file]"
    sys.exit(0)

labelled_ex_file = sys.argv[1]
ex_file = h5py.File(labelled_ex_file, 'r+')



images_folder = sys.argv[1]
if images_folder[-1] is not '/':
    images_folder = images_folder + "/"
print "Loading images from: " , images_folder

image_filenames = sorted(os.listdir(images_folder))
task_index = 0

num_steps = 1
first_task = get_task(0, False)
while get_task(num_steps, False) == first_task:
    num_steps += 1

print "Number of steps per knot tying attempt: ", num_steps

results = {}
root = Tk()
root.wm_title("Not a Knot?")
app = Application(master=root)
app.focus_set()
app.bind('y', lambda event: app.record_yes())
app.bind('n', lambda event: app.record_no())
app.bind('u', lambda event: app.undo_record())
app.bind('f', lambda event: app.note_failure())

failures = []  # List of interesting failures
panels = {}
imgs = {}

top_frame = Frame(root)
top_frame.pack()

im_size = (root.winfo_screenwidth()-200) / num_steps

for i in range(num_steps):
    im = Image.open(images_folder + image_filenames[i])
    im.thumbnail((im_size, im_size), Image.ANTIALIAS)
    imgs[i] = ImageTk.PhotoImage(im)
    panels[i] = Label(top_frame, image = imgs[i])
    panels[i].pack(side = "left", fill = "both", expand = "yes")

app.mainloop()

accuracy = float(sum(results.values())) / len(results.values())
print "Accuracy: ", accuracy

output_file = sys.argv[2]
print "Writing failures to file ", output_file
with open (output_file, 'w') as f:
    for k in failures:
        f.write("%s\n" % k)
f.close()

if len(sys.argv) > 3:
    output_file = sys.argv[3]
    print "Writing results to file ", output_file
    with open (output_file, 'w') as f:
        for k in sorted(results.keys()):
            f.write("%s\t%s\n" % (k, results[k]))
    f.close()

root.destroy()
