# --------------------------------------------------------------------
# The ElementTree toolkit is
# Copyright (c) 1999-2004 by Fredrik Lundh
# --------------------------------------------------------------------

"""
A set of HTML generator tags for building HTML documents.

Usage::

    >>> from lxml.html.builder import *
    >>> html = HTML(
    ...            HEAD( TITLE("Hello World") ),
    ...            BODY( CLASS("main"),
    ...                  H1("Hello World !")
    ...            )
    ...        )

    >>> import lxml.etree
    >>> print lxml.etree.tostring(html, pretty_print=True)
    <html>
      <head>
        <title>Hello World</title>
      </head>
      <body class="main">
        <h1>Hello World !</h1>
      </body>
    </html>

"""

from lxml.builder import ElementMaker
from lxml.html import html_parser

E = ElementMaker(makeelement=html_parser.makeelement)

# elements
A = E.a  #: anchor
ABBR = E.abbr  #: abbreviated form (e.g., WWW, HTTP, etc.)
ACRONYM = E.acronym  #: 
ADDRESS = E.address  #: information on author
APPLET = E.applet  #: Java applet (DEPRECATED)
AREA = E.area  #: client-side image map area
B = E.b  #: bold text style
BASE = E.base  #: document base URI
BASEFONT = E.basefont  #: base font size (DEPRECATED)
BDO = E.bdo  #: I18N BiDi over-ride
BIG = E.big  #: large text style
BLOCKQUOTE = E.blockquote  #: long quotation
BODY = E.body  #: document body
BR = E.br  #: forced line break
BUTTON = E.button  #: push button
CAPTION = E.caption  #: table caption
CENTER = E.center  #: shorthand for DIV align=center (DEPRECATED)
CITE = E.cite  #: citation
CODE = E.code  #: computer code fragment
COL = E.col  #: table column
COLGROUP = E.colgroup  #: table column group
DD = E.dd  #: definition description
DEL = getattr(E, 'del')  #: deleted text
DFN = E.dfn  #: instance definition
DIR = E.dir  #: directory list (DEPRECATED)
DIV = E.div  #: generic language/style container
DL = E.dl  #: definition list
DT = E.dt  #: definition term
EM = E.em  #: emphasis
FIELDSET = E.fieldset  #: form control group
FONT = E.font  #: local change to font (DEPRECATED)
FORM = E.form  #: interactive form
FRAME = E.frame  #: subwindow
FRAMESET = E.frameset  #: window subdivision
H1 = E.h1  #: heading
H2 = E.h2  #: heading
H3 = E.h3  #: heading
H4 = E.h4  #: heading
H5 = E.h5  #: heading
H6 = E.h6  #: heading
HEAD = E.head  #: document head
HR = E.hr  #: horizontal rule
HTML = E.html  #: document root element
I = E.i  #: italic text style
IFRAME = E.iframe  #: inline subwindow
IMG = E.img  #: Embedded image
INPUT = E.input  #: form control
INS = E.ins  #: inserted text
ISINDEX = E.isindex  #: single line prompt (DEPRECATED)
KBD = E.kbd  #: text to be entered by the user
LABEL = E.label  #: form field label text
LEGEND = E.legend  #: fieldset legend
LI = E.li  #: list item
LINK = E.link  #: a media-independent link
MAP = E.map  #: client-side image map
MENU = E.menu  #: menu list (DEPRECATED)
META = E.meta  #: generic metainformation
NOFRAMES = E.noframes  #: alternate content container for non frame-based rendering
NOSCRIPT = E.noscript  #: alternate content container for non script-based rendering
OBJECT = E.object  #: generic embedded object
OL = E.ol  #: ordered list
OPTGROUP = E.optgroup  #: option group
OPTION = E.option  #: selectable choice
P = E.p  #: paragraph
PARAM = E.param  #: named property value
PRE = E.pre  #: preformatted text
Q = E.q  #: short inline quotation
S = E.s  #: strike-through text style (DEPRECATED)
SAMP = E.samp  #: sample program output, scripts, etc.
SCRIPT = E.script  #: script statements
SELECT = E.select  #: option selector
SMALL = E.small  #: small text style
SPAN = E.span  #: generic language/style container
STRIKE = E.strike  #: strike-through text (DEPRECATED)
STRONG = E.strong  #: strong emphasis
STYLE = E.style  #: style info
SUB = E.sub  #: subscript
SUP = E.sup  #: superscript
TABLE = E.table  #: 
TBODY = E.tbody  #: table body
TD = E.td  #: table data cell
TEXTAREA = E.textarea  #: multi-line text field
TFOOT = E.tfoot  #: table footer
TH = E.th  #: table header cell
THEAD = E.thead  #: table header
TITLE = E.title  #: document title
TR = E.tr  #: table row
TT = E.tt  #: teletype or monospaced text style
U = E.u  #: underlined text style (DEPRECATED)
UL = E.ul  #: unordered list
VAR = E.var  #: instance of a variable or program argument

# attributes (only reserved words are included here)
ATTR = dict
def CLASS(v): return {'class': v}
def FOR(v): return {'for': v}
