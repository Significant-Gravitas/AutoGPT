       function t0(value)
         character value
         character t0
         t0 = value
       end
       function t1(value)
         character*1 value
         character*1 t1
         t1 = value
       end
       function t5(value)
         character*5 value
         character*5 t5
         t5 = value
       end
       function ts(value)
         character*(*) value
         character*(*) ts
         ts = value
       end

       subroutine s0(t0,value)
         character value
         character t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s1(t1,value)
         character*1 value
         character*1 t1
cf2py    intent(out) t1
         t1 = value
       end
       subroutine s5(t5,value)
         character*5 value
         character*5 t5
cf2py    intent(out) t5
         t5 = value
       end
       subroutine ss(ts,value)
         character*(*) value
         character*10 ts
cf2py    intent(out) ts
         ts = value
       end
