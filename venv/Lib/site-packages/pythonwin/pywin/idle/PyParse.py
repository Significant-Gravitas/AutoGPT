import re
import string
import sys

# Reason last stmt is continued (or C_NONE if it's not).
C_NONE, C_BACKSLASH, C_STRING, C_BRACKET = list(range(4))

if 0:  # for throwaway debugging output

    def dump(*stuff):
        sys.__stdout__.write(" ".join(map(str, stuff)) + "\n")


# Find what looks like the start of a popular stmt.

_synchre = re.compile(
    r"""
    ^
    [ \t]*
    (?: if
    |   for
    |   while
    |   else
    |   def
    |   return
    |   assert
    |   break
    |   class
    |   continue
    |   elif
    |   try
    |   except
    |   raise
    |   import
    )
    \b
""",
    re.VERBOSE | re.MULTILINE,
).search

# Match blank line or non-indenting comment line.

_junkre = re.compile(
    r"""
    [ \t]*
    (?: \# \S .* )?
    \n
""",
    re.VERBOSE,
).match

# Match any flavor of string; the terminating quote is optional
# so that we're robust in the face of incomplete program text.

_match_stringre = re.compile(
    r"""
    \""" [^"\\]* (?:
                     (?: \\. | "(?!"") )
                     [^"\\]*
                 )*
    (?: \""" )?

|   " [^"\\\n]* (?: \\. [^"\\\n]* )* "?

|   ''' [^'\\]* (?:
                   (?: \\. | '(?!'') )
                   [^'\\]*
                )*
    (?: ''' )?

|   ' [^'\\\n]* (?: \\. [^'\\\n]* )* '?
""",
    re.VERBOSE | re.DOTALL,
).match

# Match a line that starts with something interesting;
# used to find the first item of a bracket structure.

_itemre = re.compile(
    r"""
    [ \t]*
    [^\s#\\]    # if we match, m.end()-1 is the interesting char
""",
    re.VERBOSE,
).match

# Match start of stmts that should be followed by a dedent.

_closere = re.compile(
    r"""
    \s*
    (?: return
    |   break
    |   continue
    |   raise
    |   pass
    )
    \b
""",
    re.VERBOSE,
).match

# Chew up non-special chars as quickly as possible.  If match is
# successful, m.end() less 1 is the index of the last boring char
# matched.  If match is unsuccessful, the string starts with an
# interesting char.

_chew_ordinaryre = re.compile(
    r"""
    [^[\](){}#'"\\]+
""",
    re.VERBOSE,
).match

# Build translation table to map uninteresting chars to "x", open
# brackets to "(", and close brackets to ")".

_tran = ["x"] * 256
for ch in "({[":
    _tran[ord(ch)] = "("
for ch in ")}]":
    _tran[ord(ch)] = ")"
for ch in "\"'\\\n#":
    _tran[ord(ch)] = ch
# We are called with unicode strings, and str.translate is one of the few
# py2k functions which can't 'do the right thing' - so take care to ensure
# _tran is full of unicode...
_tran = "".join(_tran)
del ch


class Parser:
    def __init__(self, indentwidth, tabwidth):
        self.indentwidth = indentwidth
        self.tabwidth = tabwidth

    def set_str(self, str):
        assert len(str) == 0 or str[-1] == "\n", "Oops - have str %r" % (str,)
        self.str = str
        self.study_level = 0

    # Return index of a good place to begin parsing, as close to the
    # end of the string as possible.  This will be the start of some
    # popular stmt like "if" or "def".  Return None if none found:
    # the caller should pass more prior context then, if possible, or
    # if not (the entire program text up until the point of interest
    # has already been tried) pass 0 to set_lo.
    #
    # This will be reliable iff given a reliable is_char_in_string
    # function, meaning that when it says "no", it's absolutely
    # guaranteed that the char is not in a string.
    #
    # Ack, hack: in the shell window this kills us, because there's
    # no way to tell the differences between output, >>> etc and
    # user input.  Indeed, IDLE's first output line makes the rest
    # look like it's in an unclosed paren!:
    # Python 1.5.2 (#0, Apr 13 1999, ...

    def find_good_parse_start(self, use_ps1, is_char_in_string=None):
        str, pos = self.str, None
        if use_ps1:
            # shell window
            ps1 = "\n" + sys.ps1
            i = str.rfind(ps1)
            if i >= 0:
                pos = i + len(ps1)
                # make it look like there's a newline instead
                # of ps1 at the start -- hacking here once avoids
                # repeated hackery later
                self.str = str[: pos - 1] + "\n" + str[pos:]
            return pos

        # File window -- real work.
        if not is_char_in_string:
            # no clue -- make the caller pass everything
            return None

        # Peek back from the end for a good place to start,
        # but don't try too often; pos will be left None, or
        # bumped to a legitimate synch point.
        limit = len(str)
        for tries in range(5):
            i = str.rfind(":\n", 0, limit)
            if i < 0:
                break
            i = str.rfind("\n", 0, i) + 1  # start of colon line
            m = _synchre(str, i, limit)
            if m and not is_char_in_string(m.start()):
                pos = m.start()
                break
            limit = i
        if pos is None:
            # Nothing looks like a block-opener, or stuff does
            # but is_char_in_string keeps returning true; most likely
            # we're in or near a giant string, the colorizer hasn't
            # caught up enough to be helpful, or there simply *aren't*
            # any interesting stmts.  In any of these cases we're
            # going to have to parse the whole thing to be sure, so
            # give it one last try from the start, but stop wasting
            # time here regardless of the outcome.
            m = _synchre(str)
            if m and not is_char_in_string(m.start()):
                pos = m.start()
            return pos

        # Peeking back worked; look forward until _synchre no longer
        # matches.
        i = pos + 1
        while 1:
            m = _synchre(str, i)
            if m:
                s, i = m.span()
                if not is_char_in_string(s):
                    pos = s
            else:
                break
        return pos

    # Throw away the start of the string.  Intended to be called with
    # find_good_parse_start's result.

    def set_lo(self, lo):
        assert lo == 0 or self.str[lo - 1] == "\n"
        if lo > 0:
            self.str = self.str[lo:]

    # As quickly as humanly possible <wink>, find the line numbers (0-
    # based) of the non-continuation lines.
    # Creates self.{goodlines, continuation}.

    def _study1(self):
        if self.study_level >= 1:
            return
        self.study_level = 1

        # Map all uninteresting characters to "x", all open brackets
        # to "(", all close brackets to ")", then collapse runs of
        # uninteresting characters.  This can cut the number of chars
        # by a factor of 10-40, and so greatly speed the following loop.
        str = self.str
        str = str.translate(_tran)
        str = str.replace("xxxxxxxx", "x")
        str = str.replace("xxxx", "x")
        str = str.replace("xx", "x")
        str = str.replace("xx", "x")
        str = str.replace("\nx", "\n")
        # note that replacing x\n with \n would be incorrect, because
        # x may be preceded by a backslash

        # March over the squashed version of the program, accumulating
        # the line numbers of non-continued stmts, and determining
        # whether & why the last stmt is a continuation.
        continuation = C_NONE
        level = lno = 0  # level is nesting level; lno is line number
        self.goodlines = goodlines = [0]
        push_good = goodlines.append
        i, n = 0, len(str)
        while i < n:
            ch = str[i]
            i = i + 1

            # cases are checked in decreasing order of frequency
            if ch == "x":
                continue

            if ch == "\n":
                lno = lno + 1
                if level == 0:
                    push_good(lno)
                    # else we're in an unclosed bracket structure
                continue

            if ch == "(":
                level = level + 1
                continue

            if ch == ")":
                if level:
                    level = level - 1
                    # else the program is invalid, but we can't complain
                continue

            if ch == '"' or ch == "'":
                # consume the string
                quote = ch
                if str[i - 1 : i + 2] == quote * 3:
                    quote = quote * 3
                w = len(quote) - 1
                i = i + w
                while i < n:
                    ch = str[i]
                    i = i + 1

                    if ch == "x":
                        continue

                    if str[i - 1 : i + w] == quote:
                        i = i + w
                        break

                    if ch == "\n":
                        lno = lno + 1
                        if w == 0:
                            # unterminated single-quoted string
                            if level == 0:
                                push_good(lno)
                            break
                        continue

                    if ch == "\\":
                        assert i < n
                        if str[i] == "\n":
                            lno = lno + 1
                        i = i + 1
                        continue

                    # else comment char or paren inside string

                else:
                    # didn't break out of the loop, so we're still
                    # inside a string
                    continuation = C_STRING
                continue  # with outer loop

            if ch == "#":
                # consume the comment
                i = str.find("\n", i)
                assert i >= 0
                continue

            assert ch == "\\"
            assert i < n
            if str[i] == "\n":
                lno = lno + 1
                if i + 1 == n:
                    continuation = C_BACKSLASH
            i = i + 1

        # The last stmt may be continued for all 3 reasons.
        # String continuation takes precedence over bracket
        # continuation, which beats backslash continuation.
        if continuation != C_STRING and level > 0:
            continuation = C_BRACKET
        self.continuation = continuation

        # Push the final line number as a sentinel value, regardless of
        # whether it's continued.
        assert (continuation == C_NONE) == (goodlines[-1] == lno)
        if goodlines[-1] != lno:
            push_good(lno)

    def get_continuation_type(self):
        self._study1()
        return self.continuation

    # study1 was sufficient to determine the continuation status,
    # but doing more requires looking at every character.  study2
    # does this for the last interesting statement in the block.
    # Creates:
    #     self.stmt_start, stmt_end
    #         slice indices of last interesting stmt
    #     self.lastch
    #         last non-whitespace character before optional trailing
    #         comment
    #     self.lastopenbracketpos
    #         if continuation is C_BRACKET, index of last open bracket

    def _study2(self):
        _ws = string.whitespace
        if self.study_level >= 2:
            return
        self._study1()
        self.study_level = 2

        # Set p and q to slice indices of last interesting stmt.
        str, goodlines = self.str, self.goodlines
        i = len(goodlines) - 1
        p = len(str)  # index of newest line
        while i:
            assert p
            # p is the index of the stmt at line number goodlines[i].
            # Move p back to the stmt at line number goodlines[i-1].
            q = p
            for nothing in range(goodlines[i - 1], goodlines[i]):
                # tricky: sets p to 0 if no preceding newline
                p = str.rfind("\n", 0, p - 1) + 1
            # The stmt str[p:q] isn't a continuation, but may be blank
            # or a non-indenting comment line.
            if _junkre(str, p):
                i = i - 1
            else:
                break
        if i == 0:
            # nothing but junk!
            assert p == 0
            q = p
        self.stmt_start, self.stmt_end = p, q

        # Analyze this stmt, to find the last open bracket (if any)
        # and last interesting character (if any).
        lastch = ""
        stack = []  # stack of open bracket indices
        push_stack = stack.append
        while p < q:
            # suck up all except ()[]{}'"#\\
            m = _chew_ordinaryre(str, p, q)
            if m:
                # we skipped at least one boring char
                newp = m.end()
                # back up over totally boring whitespace
                i = newp - 1  # index of last boring char
                while i >= p and str[i] in " \t\n":
                    i = i - 1
                if i >= p:
                    lastch = str[i]
                p = newp
                if p >= q:
                    break

            ch = str[p]

            if ch in "([{":
                push_stack(p)
                lastch = ch
                p = p + 1
                continue

            if ch in ")]}":
                if stack:
                    del stack[-1]
                lastch = ch
                p = p + 1
                continue

            if ch == '"' or ch == "'":
                # consume string
                # Note that study1 did this with a Python loop, but
                # we use a regexp here; the reason is speed in both
                # cases; the string may be huge, but study1 pre-squashed
                # strings to a couple of characters per line.  study1
                # also needed to keep track of newlines, and we don't
                # have to.
                lastch = ch
                p = _match_stringre(str, p, q).end()
                continue

            if ch == "#":
                # consume comment and trailing newline
                p = str.find("\n", p, q) + 1
                assert p > 0
                continue

            assert ch == "\\"
            p = p + 1  # beyond backslash
            assert p < q
            if str[p] != "\n":
                # the program is invalid, but can't complain
                lastch = ch + str[p]
            p = p + 1  # beyond escaped char

        # end while p < q:

        self.lastch = lastch
        if stack:
            self.lastopenbracketpos = stack[-1]

    # Assuming continuation is C_BRACKET, return the number
    # of spaces the next line should be indented.

    def compute_bracket_indent(self):
        self._study2()
        assert self.continuation == C_BRACKET
        j = self.lastopenbracketpos
        str = self.str
        n = len(str)
        origi = i = str.rfind("\n", 0, j) + 1
        j = j + 1  # one beyond open bracket
        # find first list item; set i to start of its line
        while j < n:
            m = _itemre(str, j)
            if m:
                j = m.end() - 1  # index of first interesting char
                extra = 0
                break
            else:
                # this line is junk; advance to next line
                i = j = str.find("\n", j) + 1
        else:
            # nothing interesting follows the bracket;
            # reproduce the bracket line's indentation + a level
            j = i = origi
            while str[j] in " \t":
                j = j + 1
            extra = self.indentwidth
        return len(str[i:j].expandtabs(self.tabwidth)) + extra

    # Return number of physical lines in last stmt (whether or not
    # it's an interesting stmt!  this is intended to be called when
    # continuation is C_BACKSLASH).

    def get_num_lines_in_stmt(self):
        self._study1()
        goodlines = self.goodlines
        return goodlines[-1] - goodlines[-2]

    # Assuming continuation is C_BACKSLASH, return the number of spaces
    # the next line should be indented.  Also assuming the new line is
    # the first one following the initial line of the stmt.

    def compute_backslash_indent(self):
        self._study2()
        assert self.continuation == C_BACKSLASH
        str = self.str
        i = self.stmt_start
        while str[i] in " \t":
            i = i + 1
        startpos = i

        # See whether the initial line starts an assignment stmt; i.e.,
        # look for an = operator
        endpos = str.find("\n", startpos) + 1
        found = level = 0
        while i < endpos:
            ch = str[i]
            if ch in "([{":
                level = level + 1
                i = i + 1
            elif ch in ")]}":
                if level:
                    level = level - 1
                i = i + 1
            elif ch == '"' or ch == "'":
                i = _match_stringre(str, i, endpos).end()
            elif ch == "#":
                break
            elif (
                level == 0
                and ch == "="
                and (i == 0 or str[i - 1] not in "=<>!")
                and str[i + 1] != "="
            ):
                found = 1
                break
            else:
                i = i + 1

        if found:
            # found a legit =, but it may be the last interesting
            # thing on the line
            i = i + 1  # move beyond the =
            found = re.match(r"\s*\\", str[i:endpos]) is None

        if not found:
            # oh well ... settle for moving beyond the first chunk
            # of non-whitespace chars
            i = startpos
            while str[i] not in " \t\n":
                i = i + 1

        return len(str[self.stmt_start : i].expandtabs(self.tabwidth)) + 1

    # Return the leading whitespace on the initial line of the last
    # interesting stmt.

    def get_base_indent_string(self):
        self._study2()
        i, n = self.stmt_start, self.stmt_end
        j = i
        str = self.str
        while j < n and str[j] in " \t":
            j = j + 1
        return str[i:j]

    # Did the last interesting stmt open a block?

    def is_block_opener(self):
        self._study2()
        return self.lastch == ":"

    # Did the last interesting stmt close a block?

    def is_block_closer(self):
        self._study2()
        return _closere(self.str, self.stmt_start) is not None

    # index of last open bracket ({[, or None if none
    lastopenbracketpos = None

    def get_last_open_bracket_pos(self):
        self._study2()
        return self.lastopenbracketpos
