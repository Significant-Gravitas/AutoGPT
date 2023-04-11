       function t0(value)
         logical value
         logical t0
         t0 = value
       end
       function t1(value)
         logical*1 value
         logical*1 t1
         t1 = value
       end
       function t2(value)
         logical*2 value
         logical*2 t2
         t2 = value
       end
       function t4(value)
         logical*4 value
         logical*4 t4
         t4 = value
       end
c       function t8(value)
c         logical*8 value
c         logical*8 t8
c         t8 = value
c       end

       subroutine s0(t0,value)
         logical value
         logical t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s1(t1,value)
         logical*1 value
         logical*1 t1
cf2py    intent(out) t1
         t1 = value
       end
       subroutine s2(t2,value)
         logical*2 value
         logical*2 t2
cf2py    intent(out) t2
         t2 = value
       end
       subroutine s4(t4,value)
         logical*4 value
         logical*4 t4
cf2py    intent(out) t4
         t4 = value
       end
c       subroutine s8(t8,value)
c         logical*8 value
c         logical*8 t8
cf2py    intent(out) t8
c         t8 = value
c       end
