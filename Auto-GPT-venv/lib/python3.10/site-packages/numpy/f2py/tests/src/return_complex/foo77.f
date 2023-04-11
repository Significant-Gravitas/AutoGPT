       function t0(value)
         complex value
         complex t0
         t0 = value
       end
       function t8(value)
         complex*8 value
         complex*8 t8
         t8 = value
       end
       function t16(value)
         complex*16 value
         complex*16 t16
         t16 = value
       end
       function td(value)
         double complex value
         double complex td
         td = value
       end

       subroutine s0(t0,value)
         complex value
         complex t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s8(t8,value)
         complex*8 value
         complex*8 t8
cf2py    intent(out) t8
         t8 = value
       end
       subroutine s16(t16,value)
         complex*16 value
         complex*16 t16
cf2py    intent(out) t16
         t16 = value
       end
       subroutine sd(td,value)
         double complex value
         double complex td
cf2py    intent(out) td
         td = value
       end
