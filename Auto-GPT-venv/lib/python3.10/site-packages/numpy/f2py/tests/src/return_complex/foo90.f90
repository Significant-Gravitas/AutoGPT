module f90_return_complex
  contains
       function t0(value)
         complex :: value
         complex :: t0
         t0 = value
       end function t0
       function t8(value)
         complex(kind=4) :: value
         complex(kind=4) :: t8
         t8 = value
       end function t8
       function t16(value)
         complex(kind=8) :: value
         complex(kind=8) :: t16
         t16 = value
       end function t16
       function td(value)
         double complex :: value
         double complex :: td
         td = value
       end function td

       subroutine s0(t0,value)
         complex :: value
         complex :: t0
!f2py    intent(out) t0
         t0 = value
       end subroutine s0
       subroutine s8(t8,value)
         complex(kind=4) :: value
         complex(kind=4) :: t8
!f2py    intent(out) t8
         t8 = value
       end subroutine s8
       subroutine s16(t16,value)
         complex(kind=8) :: value
         complex(kind=8) :: t16
!f2py    intent(out) t16
         t16 = value
       end subroutine s16
       subroutine sd(td,value)
         double complex :: value
         double complex :: td
!f2py    intent(out) td
         td = value
       end subroutine sd
end module f90_return_complex
