def progress_bar(done, total, pre_text="", post_text=""):
  """
  Prints a progress bar on the console.

  Args:
    done: Number of items done.
    total: Total number of items.
    pre_text: Text to print in front of the progress bar. Should have a constant number of characters
      for each call.
    post_text: Text to print after the progress bar. Should have a constant number of characters
      for each call.
  """
  done_ratio = done/total
  total_chars = 20
  blocks = round(total_chars * done_ratio)
  spaces = total_chars - blocks
  bar_text = "|" + "█"*blocks + " "*spaces + "|"
  absolute_progess_text = " {:{width}}/{:{width}}".format(done, total, width=len(str(total)))
  percentage_text = "{:3.0f}%".format(done_ratio*100)
  print(pre_text + bar_text + absolute_progess_text + " [" + percentage_text + "]" + post_text,
    end="\r" if done_ratio < 1 else "\n")
