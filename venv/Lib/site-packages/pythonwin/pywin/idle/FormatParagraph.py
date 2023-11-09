# Extension to format a paragraph

# Does basic, standard text formatting, and also understands Python
# comment blocks.  Thus, for editing Python source code, this
# extension is really only suitable for reformatting these comment
# blocks or triple-quoted strings.

# Known problems with comment reformatting:
# * If there is a selection marked, and the first line of the
#   selection is not complete, the block will probably not be detected
#   as comments, and will have the normal "text formatting" rules
#   applied.
# * If a comment block has leading whitespace that mixes tabs and
#   spaces, they will not be considered part of the same block.
# * Fancy comments, like this bulleted list, arent handled :-)

import re


class FormatParagraph:
    menudefs = [
        (
            "edit",
            [
                ("Format Paragraph", "<<format-paragraph>>"),
            ],
        )
    ]

    keydefs = {
        "<<format-paragraph>>": ["<Alt-q>"],
    }

    unix_keydefs = {
        "<<format-paragraph>>": ["<Meta-q>"],
    }

    def __init__(self, editwin):
        self.editwin = editwin

    def close(self):
        self.editwin = None

    def format_paragraph_event(self, event):
        text = self.editwin.text
        first, last = self.editwin.get_selection_indices()
        if first and last:
            data = text.get(first, last)
            comment_header = ""
        else:
            first, last, comment_header, data = find_paragraph(
                text, text.index("insert")
            )
        if comment_header:
            # Reformat the comment lines - convert to text sans header.
            lines = data.split("\n")
            lines = map(lambda st, l=len(comment_header): st[l:], lines)
            data = "\n".join(lines)
            # Reformat to 70 chars or a 20 char width, whichever is greater.
            format_width = max(70 - len(comment_header), 20)
            newdata = reformat_paragraph(data, format_width)
            # re-split and re-insert the comment header.
            newdata = newdata.split("\n")
            # If the block ends in a \n, we dont want the comment
            # prefix inserted after it. (Im not sure it makes sense to
            # reformat a comment block that isnt made of complete
            # lines, but whatever!)  Can't think of a clean soltution,
            # so we hack away
            block_suffix = ""
            if not newdata[-1]:
                block_suffix = "\n"
                newdata = newdata[:-1]
            builder = lambda item, prefix=comment_header: prefix + item
            newdata = "\n".join([builder(d) for d in newdata]) + block_suffix
        else:
            # Just a normal text format
            newdata = reformat_paragraph(data)
        text.tag_remove("sel", "1.0", "end")
        if newdata != data:
            text.mark_set("insert", first)
            text.undo_block_start()
            text.delete(first, last)
            text.insert(first, newdata)
            text.undo_block_stop()
        else:
            text.mark_set("insert", last)
        text.see("insert")


def find_paragraph(text, mark):
    lineno, col = list(map(int, mark.split(".")))
    line = text.get("%d.0" % lineno, "%d.0 lineend" % lineno)
    while text.compare("%d.0" % lineno, "<", "end") and is_all_white(line):
        lineno = lineno + 1
        line = text.get("%d.0" % lineno, "%d.0 lineend" % lineno)
    first_lineno = lineno
    comment_header = get_comment_header(line)
    comment_header_len = len(comment_header)
    while get_comment_header(line) == comment_header and not is_all_white(
        line[comment_header_len:]
    ):
        lineno = lineno + 1
        line = text.get("%d.0" % lineno, "%d.0 lineend" % lineno)
    last = "%d.0" % lineno
    # Search back to beginning of paragraph
    lineno = first_lineno - 1
    line = text.get("%d.0" % lineno, "%d.0 lineend" % lineno)
    while (
        lineno > 0
        and get_comment_header(line) == comment_header
        and not is_all_white(line[comment_header_len:])
    ):
        lineno = lineno - 1
        line = text.get("%d.0" % lineno, "%d.0 lineend" % lineno)
    first = "%d.0" % (lineno + 1)
    return first, last, comment_header, text.get(first, last)


def reformat_paragraph(data, limit=70):
    lines = data.split("\n")
    i = 0
    n = len(lines)
    while i < n and is_all_white(lines[i]):
        i = i + 1
    if i >= n:
        return data
    indent1 = get_indent(lines[i])
    if i + 1 < n and not is_all_white(lines[i + 1]):
        indent2 = get_indent(lines[i + 1])
    else:
        indent2 = indent1
    new = lines[:i]
    partial = indent1
    while i < n and not is_all_white(lines[i]):
        # XXX Should take double space after period (etc.) into account
        words = re.split("(\s+)", lines[i])
        for j in range(0, len(words), 2):
            word = words[j]
            if not word:
                continue  # Can happen when line ends in whitespace
            if len((partial + word).expandtabs()) > limit and partial != indent1:
                new.append(partial.rstrip())
                partial = indent2
            partial = partial + word + " "
            if j + 1 < len(words) and words[j + 1] != " ":
                partial = partial + " "
        i = i + 1
    new.append(partial.rstrip())
    # XXX Should reformat remaining paragraphs as well
    new.extend(lines[i:])
    return "\n".join(new)


def is_all_white(line):
    return re.match(r"^\s*$", line) is not None


def get_indent(line):
    return re.match(r"^(\s*)", line).group()


def get_comment_header(line):
    m = re.match(r"^(\s*#*)", line)
    if m is None:
        return ""
    return m.group(1)
