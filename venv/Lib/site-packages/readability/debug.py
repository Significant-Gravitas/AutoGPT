import re


# FIXME: use with caution, can leak memory
uids = {}
uids_document = None


def describe_node(node):
    global uids
    if node is None:
        return ""
    if not hasattr(node, "tag"):
        return "[%s]" % type(node)
    name = node.tag
    if node.get("id", ""):
        name += "#" + node.get("id")
    if node.get("class", "").strip():
        name += "." + ".".join(node.get("class").split())
    if name[:4] in ["div#", "div."]:
        name = name[3:]
    if name in ["tr", "td", "div", "p"]:
        uid = uids.get(node)
        if uid is None:
            uid = uids[node] = len(uids) + 1
        name += "{%02d}" % uid
    return name


def describe(node, depth=1):
    global uids, uids_document
    doc = node.getroottree().getroot()
    if doc != uids_document:
        uids = {}
        uids_document = doc

    # return repr(NodeRepr(node))
    parent = ""
    if depth and node.getparent() is not None:
        parent = describe(node.getparent(), depth=depth - 1) + ">"
    return parent + describe_node(node)


RE_COLLAPSE_WHITESPACES = re.compile(r"\s+", re.U)


def text_content(elem, length=40):
    content = RE_COLLAPSE_WHITESPACES.sub(" ", elem.text_content().replace("\r", ""))
    if len(content) < length:
        return content
    return content[:length] + "..."
