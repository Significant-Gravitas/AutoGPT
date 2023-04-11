module f90_return_integer
  contains
       function t0(value)
         integer :: value
         integer :: t0
         t0 = value
       end function t0
       function t1(value)
         integer(kind=1) :: value
         integer(kind=1) :: t1
         t1 = value
       end function t1
       function t2(value)
         integer(kind=2) :: value
         integer(kind=2) :: t2
         t2 = value
       end function t2
       function t4(value)
         integer(kind=4) :: value
         integer(kind=4) :: t4
         t4 = value
       end function t4
       function t8(value)
         integer(kind=8) :: value
         integer(kind=8) :: t8
         t8 = value
       end function t8

       subroutine s0(t0,value)
         integer :: value
         integer :: t0
!f2py    intent(out) t0
         t0 = value
       end subroutine s0
       subroutine s1(t1,value)
         integer(kind=1) :: value
         integer(kind=1) :: t1
!f2py    intent(out) t1
         t1 = value
       end subroutine s1
       subroutine s2(t2,value)
         integer(kind=2) :: value
         integer(kind=2) :: t2
!f2py    intent(out) t2
         t2 = value
       end subroutine s2
       subroutine s4(t4,value)
         integer(kind=4) :: value
         integer(kind=4) :: t4
!f2py    intent(out) t4
         t4 = value
       end subroutine s4
       subroutine s8(t8,value)
         integer(kind=8) :: value
         integer(kind=8) :: t8
!f2py    intent(out) t8
         t8 = value
       end subroutine s8
end module f90_return_integer
