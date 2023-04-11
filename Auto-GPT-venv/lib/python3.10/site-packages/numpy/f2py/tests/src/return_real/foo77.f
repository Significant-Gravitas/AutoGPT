       function t0(value)
         real value
         real t0
         t0 = value
       end
       function t4(value)
         real*4 value
         real*4 t4
         t4 = value
       end
       function t8(value)
         real*8 value
         real*8 t8
         t8 = value
       end
       function td(value)
         double precision value
         double precision td
         td = value
       end

       subroutine s0(t0,value)
         real value
         real t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s4(t4,value)
         real*4 value
         real*4 t4
cf2py    intent(out) t4
         t4 = value
       end
       subroutine s8(t8,value)
         real*8 value
         real*8 t8
cf2py    intent(out) t8
         t8 = value
       end
       subroutine sd(td,value)
         double precision value
         double precision td
cf2py    intent(out) td
         td = value
       end
