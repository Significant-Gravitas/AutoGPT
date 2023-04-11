      subroutine func1(n, x, res)
        use, intrinsic :: iso_fortran_env, only: int64, real64
        implicit none
        integer(int64), intent(in) :: n
        real(real64), intent(in) :: x(n)
        real(real64), intent(out) :: res
Cf2py   intent(hide) :: n
        res = sum(x)
      end
