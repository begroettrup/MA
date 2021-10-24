import os

import argparse

import numpy as np
from interfaces.command_line import progress_bar

from PIL import Image
from PIL import ImageChops

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("path", type=Path, help="Path to check for duplicates.")

args = parser.parse_args()

path = args.path

image_names = list(path.iterdir())

sizes = np.array(list(map(os.path.getsize,image_names)))

sort_args = sizes.argsort()

sorted_sizes = sizes[sort_args]
# get all elements that are equal to successor
equal_to_successor = sorted_sizes[:-1] == sorted_sizes[1:]

visited = np.full(len(image_names), False)

equal_size_ids = np.flatnonzero(equal_to_successor)

checked = 0
problems = 0

def visit(i):
  visited[i] = True
  global checked
  if equal_to_successor[i]:
    checked += 1

progress_bar(0, len(equal_size_ids), pre_text=" Checking for Duplicates ")

for i in equal_size_ids:
  if visited[i]:
    continue

  visit(i)
  k = i + 1

  this_fn = image_names[sort_args[i]]
  sames = [this_fn]
  this_img = Image.open(image_names[sort_args[i]]).convert("RGB")
  
  while k<len(sorted_sizes) and sorted_sizes[k] == sorted_sizes[i]:
    if not visited[k]:
      other_fn = image_names[sort_args[k]]
      other_img = Image.open(other_fn).convert("RGB")
      
      diff = ImageChops.difference(this_img, other_img)

      if not diff.getbbox():
        visit(k)
        sames.append(other_fn)

    k += 1

  progress_bar(checked, len(equal_size_ids), pre_text=" Checking for Duplicates ")

  if len(sames) > 1:
    problems += 1
    print()
    print("Duplicate set detected!")
    for s in sames:
      print(" ->", s)

print("Checked", checked, "out of", len(equal_size_ids), "potential collisions.")

if problems == 0:
  print("No duplicates detected!")
else:
  print("Detected", problems, "sets of duplicates!")
