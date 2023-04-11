#!/usr/bin/env python3
"""
cpuinfo

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy (BSD style) license.  See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Pearu Peterson

"""
__all__ = ['cpu']

import os
import platform
import re
import sys
import types
import warnings

from subprocess import getstatusoutput


def getoutput(cmd, successful_status=(0,), stacklevel=1):
    try:
        status, output = getstatusoutput(cmd)
    except OSError as e:
        warnings.warn(str(e), UserWarning, stacklevel=stacklevel)
        return False, ""
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) in successful_status:
        return True, output
    return False, output

def command_info(successful_status=(0,), stacklevel=1, **kw):
    info = {}
    for key in kw:
        ok, output = getoutput(kw[key], successful_status=successful_status,
                               stacklevel=stacklevel+1)
        if ok:
            info[key] = output.strip()
    return info

def command_by_line(cmd, successful_status=(0,), stacklevel=1):
    ok, output = getoutput(cmd, successful_status=successful_status,
                           stacklevel=stacklevel+1)
    if not ok:
        return
    for line in output.splitlines():
        yield line.strip()

def key_value_from_command(cmd, sep, successful_status=(0,),
                           stacklevel=1):
    d = {}
    for line in command_by_line(cmd, successful_status=successful_status,
                                stacklevel=stacklevel+1):
        l = [s.strip() for s in line.split(sep, 1)]
        if len(l) == 2:
            d[l[0]] = l[1]
    return d

class CPUInfoBase:
    """Holds CPU information and provides methods for requiring
    the availability of various CPU features.
    """

    def _try_call(self, func):
        try:
            return func()
        except Exception:
            pass

    def __getattr__(self, name):
        if not name.startswith('_'):
            if hasattr(self, '_'+name):
                attr = getattr(self, '_'+name)
                if isinstance(attr, types.MethodType):
                    return lambda func=self._try_call,attr=attr : func(attr)
            else:
                return lambda : None
        raise AttributeError(name)

    def _getNCPUs(self):
        return 1

    def __get_nbits(self):
        abits = platform.architecture()[0]
        nbits = re.compile(r'(\d+)bit').search(abits).group(1)
        return nbits

    def _is_32bit(self):
        return self.__get_nbits() == '32'

    def _is_64bit(self):
        return self.__get_nbits() == '64'

class LinuxCPUInfo(CPUInfoBase):

    info = None

    def __init__(self):
        if self.info is not None:
            return
        info = [ {} ]
        ok, output = getoutput('uname -m')
        if ok:
            info[0]['uname_m'] = output.strip()
        try:
            fo = open('/proc/cpuinfo')
        except OSError as e:
            warnings.warn(str(e), UserWarning, stacklevel=2)
        else:
            for line in fo:
                name_value = [s.strip() for s in line.split(':', 1)]
                if len(name_value) != 2:
                    continue
                name, value = name_value
                if not info or name in info[-1]: # next processor
                    info.append({})
                info[-1][name] = value
            fo.close()
        self.__class__.info = info

    def _not_impl(self): pass

    # Athlon

    def _is_AMD(self):
        return self.info[0]['vendor_id']=='AuthenticAMD'

    def _is_AthlonK6_2(self):
        return self._is_AMD() and self.info[0]['model'] == '2'

    def _is_AthlonK6_3(self):
        return self._is_AMD() and self.info[0]['model'] == '3'

    def _is_AthlonK6(self):
        return re.match(r'.*?AMD-K6', self.info[0]['model name']) is not None

    def _is_AthlonK7(self):
        return re.match(r'.*?AMD-K7', self.info[0]['model name']) is not None

    def _is_AthlonMP(self):
        return re.match(r'.*?Athlon\(tm\) MP\b',
                        self.info[0]['model name']) is not None

    def _is_AMD64(self):
        return self.is_AMD() and self.info[0]['family'] == '15'

    def _is_Athlon64(self):
        return re.match(r'.*?Athlon\(tm\) 64\b',
                        self.info[0]['model name']) is not None

    def _is_AthlonHX(self):
        return re.match(r'.*?Athlon HX\b',
                        self.info[0]['model name']) is not None

    def _is_Opteron(self):
        return re.match(r'.*?Opteron\b',
                        self.info[0]['model name']) is not None

    def _is_Hammer(self):
        return re.match(r'.*?Hammer\b',
                        self.info[0]['model name']) is not None

    # Alpha

    def _is_Alpha(self):
        return self.info[0]['cpu']=='Alpha'

    def _is_EV4(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'EV4'

    def _is_EV5(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'EV5'

    def _is_EV56(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'EV56'

    def _is_PCA56(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'PCA56'

    # Intel

    #XXX
    _is_i386 = _not_impl

    def _is_Intel(self):
        return self.info[0]['vendor_id']=='GenuineIntel'

    def _is_i486(self):
        return self.info[0]['cpu']=='i486'

    def _is_i586(self):
        return self.is_Intel() and self.info[0]['cpu family'] == '5'

    def _is_i686(self):
        return self.is_Intel() and self.info[0]['cpu family'] == '6'

    def _is_Celeron(self):
        return re.match(r'.*?Celeron',
                        self.info[0]['model name']) is not None

    def _is_Pentium(self):
        return re.match(r'.*?Pentium',
                        self.info[0]['model name']) is not None

    def _is_PentiumII(self):
        return re.match(r'.*?Pentium.*?II\b',
                        self.info[0]['model name']) is not None

    def _is_PentiumPro(self):
        return re.match(r'.*?PentiumPro\b',
                        self.info[0]['model name']) is not None

    def _is_PentiumMMX(self):
        return re.match(r'.*?Pentium.*?MMX\b',
                        self.info[0]['model name']) is not None

    def _is_PentiumIII(self):
        return re.match(r'.*?Pentium.*?III\b',
                        self.info[0]['model name']) is not None

    def _is_PentiumIV(self):
        return re.match(r'.*?Pentium.*?(IV|4)\b',
                        self.info[0]['model name']) is not None

    def _is_PentiumM(self):
        return re.match(r'.*?Pentium.*?M\b',
                        self.info[0]['model name']) is not None

    def _is_Prescott(self):
        return self.is_PentiumIV() and self.has_sse3()

    def _is_Nocona(self):
        return (self.is_Intel()
                and (self.info[0]['cpu family'] == '6'
                     or self.info[0]['cpu family'] == '15')
                and (self.has_sse3() and not self.has_ssse3())
                and re.match(r'.*?\blm\b', self.info[0]['flags']) is not None)

    def _is_Core2(self):
        return (self.is_64bit() and self.is_Intel() and
                re.match(r'.*?Core\(TM\)2\b',
                         self.info[0]['model name']) is not None)

    def _is_Itanium(self):
        return re.match(r'.*?Itanium\b',
                        self.info[0]['family']) is not None

    def _is_XEON(self):
        return re.match(r'.*?XEON\b',
                        self.info[0]['model name'], re.IGNORECASE) is not None

    _is_Xeon = _is_XEON

    # Varia

    def _is_singleCPU(self):
        return len(self.info) == 1

    def _getNCPUs(self):
        return len(self.info)

    def _has_fdiv_bug(self):
        return self.info[0]['fdiv_bug']=='yes'

    def _has_f00f_bug(self):
        return self.info[0]['f00f_bug']=='yes'

    def _has_mmx(self):
        return re.match(r'.*?\bmmx\b', self.info[0]['flags']) is not None

    def _has_sse(self):
        return re.match(r'.*?\bsse\b', self.info[0]['flags']) is not None

    def _has_sse2(self):
        return re.match(r'.*?\bsse2\b', self.info[0]['flags']) is not None

    def _has_sse3(self):
        return re.match(r'.*?\bpni\b', self.info[0]['flags']) is not None

    def _has_ssse3(self):
        return re.match(r'.*?\bssse3\b', self.info[0]['flags']) is not None

    def _has_3dnow(self):
        return re.match(r'.*?\b3dnow\b', self.info[0]['flags']) is not None

    def _has_3dnowext(self):
        return re.match(r'.*?\b3dnowext\b', self.info[0]['flags']) is not None

class IRIXCPUInfo(CPUInfoBase):
    info = None

    def __init__(self):
        if self.info is not None:
            return
        info = key_value_from_command('sysconf', sep=' ',
                                      successful_status=(0, 1))
        self.__class__.info = info

    def _not_impl(self): pass

    def _is_singleCPU(self):
        return self.info.get('NUM_PROCESSORS') == '1'

    def _getNCPUs(self):
        return int(self.info.get('NUM_PROCESSORS', 1))

    def __cputype(self, n):
        return self.info.get('PROCESSORS').split()[0].lower() == 'r%s' % (n)
    def _is_r2000(self): return self.__cputype(2000)
    def _is_r3000(self): return self.__cputype(3000)
    def _is_r3900(self): return self.__cputype(3900)
    def _is_r4000(self): return self.__cputype(4000)
    def _is_r4100(self): return self.__cputype(4100)
    def _is_r4300(self): return self.__cputype(4300)
    def _is_r4400(self): return self.__cputype(4400)
    def _is_r4600(self): return self.__cputype(4600)
    def _is_r4650(self): return self.__cputype(4650)
    def _is_r5000(self): return self.__cputype(5000)
    def _is_r6000(self): return self.__cputype(6000)
    def _is_r8000(self): return self.__cputype(8000)
    def _is_r10000(self): return self.__cputype(10000)
    def _is_r12000(self): return self.__cputype(12000)
    def _is_rorion(self): return self.__cputype('orion')

    def get_ip(self):
        try: return self.info.get('MACHINE')
        except Exception: pass
    def __machine(self, n):
        return self.info.get('MACHINE').lower() == 'ip%s' % (n)
    def _is_IP19(self): return self.__machine(19)
    def _is_IP20(self): return self.__machine(20)
    def _is_IP21(self): return self.__machine(21)
    def _is_IP22(self): return self.__machine(22)
    def _is_IP22_4k(self): return self.__machine(22) and self._is_r4000()
    def _is_IP22_5k(self): return self.__machine(22)  and self._is_r5000()
    def _is_IP24(self): return self.__machine(24)
    def _is_IP25(self): return self.__machine(25)
    def _is_IP26(self): return self.__machine(26)
    def _is_IP27(self): return self.__machine(27)
    def _is_IP28(self): return self.__machine(28)
    def _is_IP30(self): return self.__machine(30)
    def _is_IP32(self): return self.__machine(32)
    def _is_IP32_5k(self): return self.__machine(32) and self._is_r5000()
    def _is_IP32_10k(self): return self.__machine(32) and self._is_r10000()


class DarwinCPUInfo(CPUInfoBase):
    info = None

    def __init__(self):
        if self.info is not None:
            return
        info = command_info(arch='arch',
                            machine='machine')
        info['sysctl_hw'] = key_value_from_command('sysctl hw', sep='=')
        self.__class__.info = info

    def _not_impl(self): pass

    def _getNCPUs(self):
        return int(self.info['sysctl_hw'].get('hw.ncpu', 1))

    def _is_Power_Macintosh(self):
        return self.info['sysctl_hw']['hw.machine']=='Power Macintosh'

    def _is_i386(self):
        return self.info['arch']=='i386'
    def _is_ppc(self):
        return self.info['arch']=='ppc'

    def __machine(self, n):
        return self.info['machine'] == 'ppc%s'%n
    def _is_ppc601(self): return self.__machine(601)
    def _is_ppc602(self): return self.__machine(602)
    def _is_ppc603(self): return self.__machine(603)
    def _is_ppc603e(self): return self.__machine('603e')
    def _is_ppc604(self): return self.__machine(604)
    def _is_ppc604e(self): return self.__machine('604e')
    def _is_ppc620(self): return self.__machine(620)
    def _is_ppc630(self): return self.__machine(630)
    def _is_ppc740(self): return self.__machine(740)
    def _is_ppc7400(self): return self.__machine(7400)
    def _is_ppc7450(self): return self.__machine(7450)
    def _is_ppc750(self): return self.__machine(750)
    def _is_ppc403(self): return self.__machine(403)
    def _is_ppc505(self): return self.__machine(505)
    def _is_ppc801(self): return self.__machine(801)
    def _is_ppc821(self): return self.__machine(821)
    def _is_ppc823(self): return self.__machine(823)
    def _is_ppc860(self): return self.__machine(860)


class SunOSCPUInfo(CPUInfoBase):

    info = None

    def __init__(self):
        if self.info is not None:
            return
        info = command_info(arch='arch',
                            mach='mach',
                            uname_i='uname_i',
                            isainfo_b='isainfo -b',
                            isainfo_n='isainfo -n',
                            )
        info['uname_X'] = key_value_from_command('uname -X', sep='=')
        for line in command_by_line('psrinfo -v 0'):
            m = re.match(r'\s*The (?P<p>[\w\d]+) processor operates at', line)
            if m:
                info['processor'] = m.group('p')
                break
        self.__class__.info = info

    def _not_impl(self): pass

    def _is_i386(self):
        return self.info['isainfo_n']=='i386'
    def _is_sparc(self):
        return self.info['isainfo_n']=='sparc'
    def _is_sparcv9(self):
        return self.info['isainfo_n']=='sparcv9'

    def _getNCPUs(self):
        return int(self.info['uname_X'].get('NumCPU', 1))

    def _is_sun4(self):
        return self.info['arch']=='sun4'

    def _is_SUNW(self):
        return re.match(r'SUNW', self.info['uname_i']) is not None
    def _is_sparcstation5(self):
        return re.match(r'.*SPARCstation-5', self.info['uname_i']) is not None
    def _is_ultra1(self):
        return re.match(r'.*Ultra-1', self.info['uname_i']) is not None
    def _is_ultra250(self):
        return re.match(r'.*Ultra-250', self.info['uname_i']) is not None
    def _is_ultra2(self):
        return re.match(r'.*Ultra-2', self.info['uname_i']) is not None
    def _is_ultra30(self):
        return re.match(r'.*Ultra-30', self.info['uname_i']) is not None
    def _is_ultra4(self):
        return re.match(r'.*Ultra-4', self.info['uname_i']) is not None
    def _is_ultra5_10(self):
        return re.match(r'.*Ultra-5_10', self.info['uname_i']) is not None
    def _is_ultra5(self):
        return re.match(r'.*Ultra-5', self.info['uname_i']) is not None
    def _is_ultra60(self):
        return re.match(r'.*Ultra-60', self.info['uname_i']) is not None
    def _is_ultra80(self):
        return re.match(r'.*Ultra-80', self.info['uname_i']) is not None
    def _is_ultraenterprice(self):
        return re.match(r'.*Ultra-Enterprise', self.info['uname_i']) is not None
    def _is_ultraenterprice10k(self):
        return re.match(r'.*Ultra-Enterprise-10000', self.info['uname_i']) is not None
    def _is_sunfire(self):
        return re.match(r'.*Sun-Fire', self.info['uname_i']) is not None
    def _is_ultra(self):
        return re.match(r'.*Ultra', self.info['uname_i']) is not None

    def _is_cpusparcv7(self):
        return self.info['processor']=='sparcv7'
    def _is_cpusparcv8(self):
        return self.info['processor']=='sparcv8'
    def _is_cpusparcv9(self):
        return self.info['processor']=='sparcv9'

class Win32CPUInfo(CPUInfoBase):

    info = None
    pkey = r"HARDWARE\DESCRIPTION\System\CentralProcessor"
    # XXX: what does the value of
    #   HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0
    # mean?

    def __init__(self):
        if self.info is not None:
            return
        info = []
        try:
            #XXX: Bad style to use so long `try:...except:...`. Fix it!
            import winreg

            prgx = re.compile(r"family\s+(?P<FML>\d+)\s+model\s+(?P<MDL>\d+)"
                              r"\s+stepping\s+(?P<STP>\d+)", re.IGNORECASE)
            chnd=winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, self.pkey)
            pnum=0
            while True:
                try:
                    proc=winreg.EnumKey(chnd, pnum)
                except winreg.error:
                    break
                else:
                    pnum+=1
                    info.append({"Processor":proc})
                    phnd=winreg.OpenKey(chnd, proc)
                    pidx=0
                    while True:
                        try:
                            name, value, vtpe=winreg.EnumValue(phnd, pidx)
                        except winreg.error:
                            break
                        else:
                            pidx=pidx+1
                            info[-1][name]=value
                            if name=="Identifier":
                                srch=prgx.search(value)
                                if srch:
                                    info[-1]["Family"]=int(srch.group("FML"))
                                    info[-1]["Model"]=int(srch.group("MDL"))
                                    info[-1]["Stepping"]=int(srch.group("STP"))
        except Exception as e:
            print(e, '(ignoring)')
        self.__class__.info = info

    def _not_impl(self): pass

    # Athlon

    def _is_AMD(self):
        return self.info[0]['VendorIdentifier']=='AuthenticAMD'

    def _is_Am486(self):
        return self.is_AMD() and self.info[0]['Family']==4

    def _is_Am5x86(self):
        return self.is_AMD() and self.info[0]['Family']==4

    def _is_AMDK5(self):
        return self.is_AMD() and self.info[0]['Family']==5 \
               and self.info[0]['Model'] in [0, 1, 2, 3]

    def _is_AMDK6(self):
        return self.is_AMD() and self.info[0]['Family']==5 \
               and self.info[0]['Model'] in [6, 7]

    def _is_AMDK6_2(self):
        return self.is_AMD() and self.info[0]['Family']==5 \
               and self.info[0]['Model']==8

    def _is_AMDK6_3(self):
        return self.is_AMD() and self.info[0]['Family']==5 \
               and self.info[0]['Model']==9

    def _is_AMDK7(self):
        return self.is_AMD() and self.info[0]['Family'] == 6

    # To reliably distinguish between the different types of AMD64 chips
    # (Athlon64, Operton, Athlon64 X2, Semperon, Turion 64, etc.) would
    # require looking at the 'brand' from cpuid

    def _is_AMD64(self):
        return self.is_AMD() and self.info[0]['Family'] == 15

    # Intel

    def _is_Intel(self):
        return self.info[0]['VendorIdentifier']=='GenuineIntel'

    def _is_i386(self):
        return self.info[0]['Family']==3

    def _is_i486(self):
        return self.info[0]['Family']==4

    def _is_i586(self):
        return self.is_Intel() and self.info[0]['Family']==5

    def _is_i686(self):
        return self.is_Intel() and self.info[0]['Family']==6

    def _is_Pentium(self):
        return self.is_Intel() and self.info[0]['Family']==5

    def _is_PentiumMMX(self):
        return self.is_Intel() and self.info[0]['Family']==5 \
               and self.info[0]['Model']==4

    def _is_PentiumPro(self):
        return self.is_Intel() and self.info[0]['Family']==6 \
               and self.info[0]['Model']==1

    def _is_PentiumII(self):
        return self.is_Intel() and self.info[0]['Family']==6 \
               and self.info[0]['Model'] in [3, 5, 6]

    def _is_PentiumIII(self):
        return self.is_Intel() and self.info[0]['Family']==6 \
               and self.info[0]['Model'] in [7, 8, 9, 10, 11]

    def _is_PentiumIV(self):
        return self.is_Intel() and self.info[0]['Family']==15

    def _is_PentiumM(self):
        return self.is_Intel() and self.info[0]['Family'] == 6 \
               and self.info[0]['Model'] in [9, 13, 14]

    def _is_Core2(self):
        return self.is_Intel() and self.info[0]['Family'] == 6 \
               and self.info[0]['Model'] in [15, 16, 17]

    # Varia

    def _is_singleCPU(self):
        return len(self.info) == 1

    def _getNCPUs(self):
        return len(self.info)

    def _has_mmx(self):
        if self.is_Intel():
            return (self.info[0]['Family']==5 and self.info[0]['Model']==4) \
                   or (self.info[0]['Family'] in [6, 15])
        elif self.is_AMD():
            return self.info[0]['Family'] in [5, 6, 15]
        else:
            return False

    def _has_sse(self):
        if self.is_Intel():
            return ((self.info[0]['Family']==6 and
                     self.info[0]['Model'] in [7, 8, 9, 10, 11])
                     or self.info[0]['Family']==15)
        elif self.is_AMD():
            return ((self.info[0]['Family']==6 and
                     self.info[0]['Model'] in [6, 7, 8, 10])
                     or self.info[0]['Family']==15)
        else:
            return False

    def _has_sse2(self):
        if self.is_Intel():
            return self.is_Pentium4() or self.is_PentiumM() \
                   or self.is_Core2()
        elif self.is_AMD():
            return self.is_AMD64()
        else:
            return False

    def _has_3dnow(self):
        return self.is_AMD() and self.info[0]['Family'] in [5, 6, 15]

    def _has_3dnowext(self):
        return self.is_AMD() and self.info[0]['Family'] in [6, 15]

if sys.platform.startswith('linux'): # variations: linux2,linux-i386 (any others?)
    cpuinfo = LinuxCPUInfo
elif sys.platform.startswith('irix'):
    cpuinfo = IRIXCPUInfo
elif sys.platform == 'darwin':
    cpuinfo = DarwinCPUInfo
elif sys.platform.startswith('sunos'):
    cpuinfo = SunOSCPUInfo
elif sys.platform.startswith('win32'):
    cpuinfo = Win32CPUInfo
elif sys.platform.startswith('cygwin'):
    cpuinfo = LinuxCPUInfo
#XXX: other OS's. Eg. use _winreg on Win32. Or os.uname on unices.
else:
    cpuinfo = CPUInfoBase

cpu = cpuinfo()

#if __name__ == "__main__":
#
#    cpu.is_blaa()
#    cpu.is_Intel()
#    cpu.is_Alpha()
#
#    print('CPU information:'),
#    for name in dir(cpuinfo):
#        if name[0]=='_' and name[1]!='_':
#            r = getattr(cpu,name[1:])()
#            if r:
#                if r!=1:
#                    print('%s=%s' %(name[1:],r))
#                else:
#                    print(name[1:]),
#    print()
