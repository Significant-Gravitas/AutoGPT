import sys

if sys.platform == "win32":

    from typing import Any

    import dns.name

    _prefer_wmi = True

    import winreg  # pylint: disable=import-error

    # Keep pylint quiet on non-windows.
    try:
        WindowsError is None  # pylint: disable=used-before-assignment
    except KeyError:
        WindowsError = Exception

    try:
        import threading
        import pythoncom  # pylint: disable=import-error
        import wmi  # pylint: disable=import-error

        _have_wmi = True
    except Exception:
        _have_wmi = False

    def _config_domain(domain):
        # Sometimes DHCP servers add a '.' prefix to the default domain, and
        # Windows just stores such values in the registry (see #687).
        # Check for this and fix it.
        if domain.startswith("."):
            domain = domain[1:]
        return dns.name.from_text(domain)

    class DnsInfo:
        def __init__(self):
            self.domain = None
            self.nameservers = []
            self.search = []

    if _have_wmi:

        class _WMIGetter(threading.Thread):
            def __init__(self):
                super().__init__()
                self.info = DnsInfo()

            def run(self):
                pythoncom.CoInitialize()
                try:
                    system = wmi.WMI()
                    for interface in system.Win32_NetworkAdapterConfiguration():
                        if interface.IPEnabled and interface.DNSDomain:
                            self.info.domain = _config_domain(interface.DNSDomain)
                            self.info.nameservers = list(interface.DNSServerSearchOrder)
                            if interface.DNSDomainSuffixSearchOrder:
                                self.info.search = [
                                    _config_domain(x)
                                    for x in interface.DNSDomainSuffixSearchOrder
                                ]
                            break
                finally:
                    pythoncom.CoUninitialize()

            def get(self):
                # We always run in a separate thread to avoid any issues with
                # the COM threading model.
                self.start()
                self.join()
                return self.info

    else:

        class _WMIGetter:  # type: ignore
            pass

    class _RegistryGetter:
        def __init__(self):
            self.info = DnsInfo()

        def _determine_split_char(self, entry):
            #
            # The windows registry irritatingly changes the list element
            # delimiter in between ' ' and ',' (and vice-versa) in various
            # versions of windows.
            #
            if entry.find(" ") >= 0:
                split_char = " "
            elif entry.find(",") >= 0:
                split_char = ","
            else:
                # probably a singleton; treat as a space-separated list.
                split_char = " "
            return split_char

        def _config_nameservers(self, nameservers):
            split_char = self._determine_split_char(nameservers)
            ns_list = nameservers.split(split_char)
            for ns in ns_list:
                if ns not in self.info.nameservers:
                    self.info.nameservers.append(ns)

        def _config_search(self, search):
            split_char = self._determine_split_char(search)
            search_list = search.split(split_char)
            for s in search_list:
                s = _config_domain(s)
                if s not in self.info.search:
                    self.info.search.append(s)

        def _config_fromkey(self, key, always_try_domain):
            try:
                servers, _ = winreg.QueryValueEx(key, "NameServer")
            except WindowsError:
                servers = None
            if servers:
                self._config_nameservers(servers)
            if servers or always_try_domain:
                try:
                    dom, _ = winreg.QueryValueEx(key, "Domain")
                    if dom:
                        self.info.domain = _config_domain(dom)
                except WindowsError:
                    pass
            else:
                try:
                    servers, _ = winreg.QueryValueEx(key, "DhcpNameServer")
                except WindowsError:
                    servers = None
                if servers:
                    self._config_nameservers(servers)
                    try:
                        dom, _ = winreg.QueryValueEx(key, "DhcpDomain")
                        if dom:
                            self.info.domain = _config_domain(dom)
                    except WindowsError:
                        pass
            try:
                search, _ = winreg.QueryValueEx(key, "SearchList")
            except WindowsError:
                search = None
            if search is None:
                try:
                    search, _ = winreg.QueryValueEx(key, "DhcpSearchList")
                except WindowsError:
                    search = None
            if search:
                self._config_search(search)

        def _is_nic_enabled(self, lm, guid):
            # Look in the Windows Registry to determine whether the network
            # interface corresponding to the given guid is enabled.
            #
            # (Code contributed by Paul Marks, thanks!)
            #
            try:
                # This hard-coded location seems to be consistent, at least
                # from Windows 2000 through Vista.
                connection_key = winreg.OpenKey(
                    lm,
                    r"SYSTEM\CurrentControlSet\Control\Network"
                    r"\{4D36E972-E325-11CE-BFC1-08002BE10318}"
                    r"\%s\Connection" % guid,
                )

                try:
                    # The PnpInstanceID points to a key inside Enum
                    (pnp_id, ttype) = winreg.QueryValueEx(
                        connection_key, "PnpInstanceID"
                    )

                    if ttype != winreg.REG_SZ:
                        raise ValueError  # pragma: no cover

                    device_key = winreg.OpenKey(
                        lm, r"SYSTEM\CurrentControlSet\Enum\%s" % pnp_id
                    )

                    try:
                        # Get ConfigFlags for this device
                        (flags, ttype) = winreg.QueryValueEx(device_key, "ConfigFlags")

                        if ttype != winreg.REG_DWORD:
                            raise ValueError  # pragma: no cover

                        # Based on experimentation, bit 0x1 indicates that the
                        # device is disabled.
                        #
                        # XXXRTH I suspect we really want to & with 0x03 so
                        # that CONFIGFLAGS_REMOVED devices are also ignored,
                        # but we're shifting to WMI as ConfigFlags is not
                        # supposed to be used.
                        return not flags & 0x1

                    finally:
                        device_key.Close()
                finally:
                    connection_key.Close()
            except Exception:  # pragma: no cover
                return False

        def get(self):
            """Extract resolver configuration from the Windows registry."""

            lm = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
            try:
                tcp_params = winreg.OpenKey(
                    lm, r"SYSTEM\CurrentControlSet" r"\Services\Tcpip\Parameters"
                )
                try:
                    self._config_fromkey(tcp_params, True)
                finally:
                    tcp_params.Close()
                interfaces = winreg.OpenKey(
                    lm,
                    r"SYSTEM\CurrentControlSet"
                    r"\Services\Tcpip\Parameters"
                    r"\Interfaces",
                )
                try:
                    i = 0
                    while True:
                        try:
                            guid = winreg.EnumKey(interfaces, i)
                            i += 1
                            key = winreg.OpenKey(interfaces, guid)
                            try:
                                if not self._is_nic_enabled(lm, guid):
                                    continue
                                self._config_fromkey(key, False)
                            finally:
                                key.Close()
                        except EnvironmentError:
                            break
                finally:
                    interfaces.Close()
            finally:
                lm.Close()
            return self.info

    _getter_class: Any
    if _have_wmi and _prefer_wmi:
        _getter_class = _WMIGetter
    else:
        _getter_class = _RegistryGetter

    def get_dns_info():
        """Extract resolver configuration."""
        getter = _getter_class()
        return getter.get()
