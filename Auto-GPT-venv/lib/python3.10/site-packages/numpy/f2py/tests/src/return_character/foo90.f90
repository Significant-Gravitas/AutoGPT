module f90_return_char
  contains
       function t0(value)
         character :: value
         character :: t0
         t0 = value
       end function t0
       function t1(value)
         character(len=1) :: value
         character(len=1) :: t1
         t1 = value
       end function t1
       function t5(value)
         character(len=5) :: value
         character(len=5) :: t5
         t5 = value
       end function t5
       function ts(value)
         character(len=*) :: value
         character(len=10) :: ts
         ts = value
       end function ts

       subroutine s0(t0,value)
         character :: value
         character :: t0
!f2py    intent(out) t0
         t0 = value
       end subroutine s0
       subroutine s1(t1,value)
         character(len=1) :: value
         character(len=1) :: t1
!f2py    intent(out) t1
         t1 = value
       end subroutine s1
       subroutine s5(t5,value)
         character(len=5) :: value
         character(len=5) :: t5
!f2py    intent(out) t5
         t5 = value
       end subroutine s5
       subroutine ss(ts,value)
         character(len=*) :: value
         character(len=10) :: ts
!f2py    intent(out) ts
         ts = value
       end subroutine ss
end module f90_return_char
