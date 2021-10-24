class Named:
  """
  An object with a name which may be None.
  """
  def __init__(self, /, name=None, *args, **kwargs):
    self.name = name
    super().__init__(*args, **kwargs)

  def personalized_message(
      self, constant_start, name_part,
      unnamed_part=None,
      constant_end=None,
      separator=" "):
    """
    Format a message adding some text if a name is present and leaving it out
    otherwise. All parts will be concatenated with the separator added between
    them.

    Args:
      constant_start: String to start the text with. None for no starting text.
      name_part: Part that will be added after constant_start and before
        constant_end if a name is present. "{name}" in this string will be
        replaced by the name.
      unnamed_part: Part that will be used if no name is present.
      constant_end: String to end the text with.
      separator: String to add between each of the concatenated parts.
    """
    whole = ""
    started = False

    def add_part(part):
      sep = separator if started else ""
      if part is not None:
        return whole + sep + part, True
      else:
        return whole, started

    whole, started = add_part(constant_start)
    if self.name is not None:
      if name_part is not None:
        name_part = name_part.format(name=self.name)
      whole, started = add_part(name_part)
    else:
      whole, started = add_part(unnamed_part)
    whole, started = add_part(constant_end)

    return whole
