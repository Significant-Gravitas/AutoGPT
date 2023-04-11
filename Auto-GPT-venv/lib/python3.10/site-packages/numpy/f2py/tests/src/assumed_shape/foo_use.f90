subroutine sum_with_use(x, res)
  use precision

  implicit none

  real(kind=rk), intent(in) :: x(:)
  real(kind=rk), intent(out) :: res

  integer :: i

  !print *, "size(x) = ", size(x)

  res = 0.0

  do i = 1, size(x)
    res = res + x(i)
  enddo

 end subroutine
