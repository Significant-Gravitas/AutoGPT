      module foo_fixed
      contains
        subroutine bar12(a)
!f2py intent(out) a
          integer a
          a = 12
        end subroutine bar12
      end module foo_fixed
