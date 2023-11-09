// Support for PythonCOM and its extensions to register the interfaces,
// gateways and IIDs it supports.
//
// The module can simply declare an array of type PyCom_InterfaceSupportInfo, then
// use the macros to populate it.
//
// See Register.cpp and AXScript.cpp for examples on its use.

#ifndef __PYTHONCOMREGISTER_H__
#define __PYTHONCOMREGISTER_H__

#include "PythonCOMServer.h"  // Need defns in this file...

typedef struct {
    const GUID *pGUID;             // The supported IID - required
    const char *interfaceName;     // Name of the interface - required
    const char *iidName;           // Name of the IID that goes into the dict. - required
    PyTypeObject *pTypeOb;         // the type object for client PyI* side - NULL for server only support.
    pfnPyGatewayConstructor ctor;  // Gateway (PyG*) interface constructor - NULL for client only support

} PyCom_InterfaceSupportInfo;

#define PYCOM_INTERFACE_IID_ONLY(ifc)                   \
    {                                                   \
        &IID_I##ifc, "I" #ifc, "IID_I" #ifc, NULL, NULL \
    }
#define PYCOM_INTERFACE_CLSID_ONLY(ifc)                        \
    {                                                          \
        &CLSID_##ifc, "CLSID_" #ifc, "CLSID_" #ifc, NULL, NULL \
    }
#define PYCOM_INTERFACE_CATID_ONLY(ifc)                        \
    {                                                          \
        &CATID_##ifc, "CATID_" #ifc, "CATID_" #ifc, NULL, NULL \
    }
#define PYCOM_INTERFACE_CLIENT_ONLY(ifc)                           \
    {                                                              \
        &IID_I##ifc, "I" #ifc, "IID_I" #ifc, &PyI##ifc::type, NULL \
    }
#define PYCOM_INTERFACE_SERVER_ONLY(ifc)                                        \
    {                                                                           \
        &IID_I##ifc, "I" #ifc, "IID_I" #ifc, NULL, GET_PYGATEWAY_CTOR(PyG##ifc) \
    }
#define PYCOM_INTERFACE_FULL(ifc)                                                          \
    {                                                                                      \
        &IID_I##ifc, "I" #ifc, "IID_I" #ifc, &PyI##ifc::type, GET_PYGATEWAY_CTOR(PyG##ifc) \
    }

// Versions that use __uuidof() to get the IID, which seems to avoid the need
// to link with a lib holding the IIDs.  Note that almost all extensions
// build with __uuidof() being the default; the build failed at 'shell' - so
// we could consider making this the default and making the 'explicit' version
// above the special case.
#define PYCOM_INTERFACE_IID_ONLY_UUIDOF(ifc)                  \
    {                                                         \
        &__uuidof(I##ifc), "I" #ifc, "IID_I" #ifc, NULL, NULL \
    }
#define PYCOM_INTERFACE_CLIENT_ONLY_UUIDOF(ifc)                          \
    {                                                                    \
        &__uuidof(I##ifc), "I" #ifc, "IID_I" #ifc, &PyI##ifc::type, NULL \
    }
#define PYCOM_INTERFACE_SERVER_ONLY_UUIDOF(ifc)                                       \
    {                                                                                 \
        &__uuidof(I##ifc), "I" #ifc, "IID_I" #ifc, NULL, GET_PYGATEWAY_CTOR(PyG##ifc) \
    }
#define PYCOM_INTERFACE_FULL_UUIDOF(ifc)                                                         \
    {                                                                                            \
        &__uuidof(I##ifc), "I" #ifc, "IID_I" #ifc, &PyI##ifc::type, GET_PYGATEWAY_CTOR(PyG##ifc) \
    }

// Prototypes for the register functions

// Register a PythonCOM extension module
PYCOM_EXPORT int PyCom_RegisterExtensionSupport(PyObject *dict, const PyCom_InterfaceSupportInfo *pInterfaces,
                                                int numEntries);

// THESE SHOULD NO LONGER BE USED.  Instead, use the functions above passing an
// array of PyCom_InterfaceSupportInfo objects.

PYCOM_EXPORT int PyCom_RegisterClientType(PyTypeObject *typeOb, const GUID *guid);

HRESULT PYCOM_EXPORT PyCom_RegisterGatewayObject(REFIID iid, pfnPyGatewayConstructor ctor, const char *interfaceName);
PYCOM_EXPORT int PyCom_IsGatewayRegistered(REFIID iid);

#endif /* __PYTHONCOMREGISTER_H__ */
