from ..helpers import nativestr


def list_to_dict(aList):
    return {nativestr(aList[i][0]): nativestr(aList[i][1]) for i in range(len(aList))}


def parse_range(response):
    """Parse range response. Used by TS.RANGE and TS.REVRANGE."""
    return [tuple((r[0], float(r[1]))) for r in response]


def parse_m_range(response):
    """Parse multi range response. Used by TS.MRANGE and TS.MREVRANGE."""
    res = []
    for item in response:
        res.append({nativestr(item[0]): [list_to_dict(item[1]), parse_range(item[2])]})
    return sorted(res, key=lambda d: list(d.keys()))


def parse_get(response):
    """Parse get response. Used by TS.GET."""
    if not response:
        return None
    return int(response[0]), float(response[1])


def parse_m_get(response):
    """Parse multi get response. Used by TS.MGET."""
    res = []
    for item in response:
        if not item[2]:
            res.append({nativestr(item[0]): [list_to_dict(item[1]), None, None]})
        else:
            res.append(
                {
                    nativestr(item[0]): [
                        list_to_dict(item[1]),
                        int(item[2][0]),
                        float(item[2][1]),
                    ]
                }
            )
    return sorted(res, key=lambda d: list(d.keys()))
