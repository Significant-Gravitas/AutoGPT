/*
 * Copyright (c) 2000-2010 Marc Alexander Lehmann <schmorp@schmorp.de>
 *
 * Redistribution and use in source and binary forms, with or without modifica-
 * tion, are permitted provided that the following conditions are met:
 *
 *   1.  Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *
 *   2.  Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MER-
 * CHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO
 * EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPE-
 * CIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTH-
 * ERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Alternatively, the contents of this file may be used under the terms of
 * the GNU General Public License ("GPL") version 2 or any later version,
 * in which case the provisions of the GPL are applicable instead of
 * the above. If you wish to allow the use of your version of this file
 * only under the terms of the GPL and not to allow others to use your
 * version of this file under the BSD license, indicate your decision
 * by deleting the provisions above and replace them with the notice
 * and other provisions required by the GPL. If you do not delete the
 * provisions above, a recipient may use your version of this file under
 * either the BSD or the GPL.
 */

#include "lzfP.h"

#if AVOID_ERRNO
# define SET_ERRNO(n)
#else
# include <errno.h>
# define SET_ERRNO(n) errno = (n)
#endif

#if USE_REP_MOVSB /* small win on amd, big loss on intel */
#if (__i386 || __amd64) && __GNUC__ >= 3
# define lzf_movsb(dst, src, len)                \
   asm ("rep movsb"                              \
        : "=D" (dst), "=S" (src), "=c" (len)     \
        :  "0" (dst),  "1" (src),  "2" (len));
#endif
#endif

#if defined(__GNUC__) && __GNUC__ >= 7
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif
size_t
lzf_decompress (const void *const in_data,  size_t in_len,
                void             *out_data, size_t out_len)
{
  u8 const *ip = (const u8 *)in_data;
  u8       *op = (u8 *)out_data;
  u8 const *const in_end  = ip + in_len;
  u8       *const out_end = op + out_len;

  while (ip < in_end)
    {
      unsigned int ctrl;
      ctrl = *ip++;

      if (ctrl < (1 << 5)) /* literal run */
        {
          ctrl++;

          if (op + ctrl > out_end)
            {
              SET_ERRNO (E2BIG);
              return 0;
            }

#if CHECK_INPUT
          if (ip + ctrl > in_end)
            {
              SET_ERRNO (EINVAL);
              return 0;
            }
#endif

#ifdef lzf_movsb
          lzf_movsb (op, ip, ctrl);
#else
          switch (ctrl)
            {
              case 32: *op++ = *ip++; case 31: *op++ = *ip++; case 30: *op++ = *ip++; case 29: *op++ = *ip++;
              case 28: *op++ = *ip++; case 27: *op++ = *ip++; case 26: *op++ = *ip++; case 25: *op++ = *ip++;
              case 24: *op++ = *ip++; case 23: *op++ = *ip++; case 22: *op++ = *ip++; case 21: *op++ = *ip++;
              case 20: *op++ = *ip++; case 19: *op++ = *ip++; case 18: *op++ = *ip++; case 17: *op++ = *ip++;
              case 16: *op++ = *ip++; case 15: *op++ = *ip++; case 14: *op++ = *ip++; case 13: *op++ = *ip++;
              case 12: *op++ = *ip++; case 11: *op++ = *ip++; case 10: *op++ = *ip++; case  9: *op++ = *ip++;
              case  8: *op++ = *ip++; case  7: *op++ = *ip++; case  6: *op++ = *ip++; case  5: *op++ = *ip++;
              case  4: *op++ = *ip++; case  3: *op++ = *ip++; case  2: *op++ = *ip++; case  1: *op++ = *ip++;
            }
#endif
        }
      else /* back reference */
        {
          unsigned int len = ctrl >> 5;

          u8 *ref = op - ((ctrl & 0x1f) << 8) - 1;

#if CHECK_INPUT
          if (ip >= in_end)
            {
              SET_ERRNO (EINVAL);
              return 0;
            }
#endif
          if (len == 7)
            {
              len += *ip++;
#if CHECK_INPUT
              if (ip >= in_end)
                {
                  SET_ERRNO (EINVAL);
                  return 0;
                }
#endif
            }

          ref -= *ip++;

          if (op + len + 2 > out_end)
            {
              SET_ERRNO (E2BIG);
              return 0;
            }

          if (ref < (u8 *)out_data)
            {
              SET_ERRNO (EINVAL);
              return 0;
            }

#ifdef lzf_movsb
          len += 2;
          lzf_movsb (op, ref, len);
#else
          switch (len)
            {
              default:
                len += 2;

                if (op >= ref + len)
                  {
                    /* disjunct areas */
                    memcpy (op, ref, len);
                    op += len;
                  }
                else
                  {
                    /* overlapping, use octte by octte copying */
                    do
                      *op++ = *ref++;
                    while (--len);
                  }

                break;

              case 9: *op++ = *ref++; /* fall-thru */
              case 8: *op++ = *ref++; /* fall-thru */
              case 7: *op++ = *ref++; /* fall-thru */
              case 6: *op++ = *ref++; /* fall-thru */
              case 5: *op++ = *ref++; /* fall-thru */
              case 4: *op++ = *ref++; /* fall-thru */
              case 3: *op++ = *ref++; /* fall-thru */
              case 2: *op++ = *ref++; /* fall-thru */
              case 1: *op++ = *ref++; /* fall-thru */
              case 0: *op++ = *ref++; /* two octets more */
                      *op++ = *ref++; /* fall-thru */
            }
#endif
        }
    }

  return op - (u8 *)out_data;
}
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif
