import termcolor

# ------ convert to colored strings (better viewed in color)------
def toRed(content): return termcolor.colored(content, "red", attrs=["bold"])
def toGreen(content): return termcolor.colored(content, "green", attrs=["bold"])
def toBlue(content): return termcolor.colored(content, "blue", attrs=["bold"])
def toCyan(content): return termcolor.colored(content, "cyan", attrs=["bold"])
def toYellow(content): return termcolor.colored(content, "yellow", attrs=["bold"])
def toMagenta(content): return termcolor.colored(content, "magenta", attrs=["bold"])


