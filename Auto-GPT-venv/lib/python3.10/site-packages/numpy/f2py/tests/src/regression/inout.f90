! Check that intent(in out) translates as intent(inout).
! The separation seems to be a common usage.
      subroutine foo(x)
          implicit none
          real(4), intent(in out) :: x
          dimension x(3)
          x(1) = x(1) + x(2) + x(3)
          return
      end
