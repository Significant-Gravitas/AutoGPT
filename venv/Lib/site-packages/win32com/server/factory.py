# Class factory utilities.
import pythoncom


def RegisterClassFactories(clsids, flags=None, clsctx=None):
    """Given a list of CLSID, create and register class factories.

    Returns a list, which should be passed to RevokeClassFactories
    """
    if flags is None:
        flags = pythoncom.REGCLS_MULTIPLEUSE | pythoncom.REGCLS_SUSPENDED
    if clsctx is None:
        clsctx = pythoncom.CLSCTX_LOCAL_SERVER
    ret = []
    for clsid in clsids:
        # Some server append '-Embedding' etc
        if clsid[0] not in ["-", "/"]:
            factory = pythoncom.MakePyFactory(clsid)
            regId = pythoncom.CoRegisterClassObject(clsid, factory, clsctx, flags)
            ret.append((factory, regId))
    return ret


def RevokeClassFactories(infos):
    for factory, revokeId in infos:
        pythoncom.CoRevokeClassObject(revokeId)
