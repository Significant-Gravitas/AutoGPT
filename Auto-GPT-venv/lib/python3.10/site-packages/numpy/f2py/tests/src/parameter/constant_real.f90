! Check that parameters are correct intercepted.
! Constants with comma separations are commonly
! used, for instance Pi = 3._dp
subroutine foo_single(x)
  implicit none
  integer, parameter :: rp = selected_real_kind(6)
  real(rp), intent(inout) :: x
  dimension x(3)
  real(rp), parameter :: three = 3._rp
  x(1) = x(1) + x(2) + x(3) * three
  return
end subroutine

subroutine foo_double(x)
  implicit none
  integer, parameter :: rp = selected_real_kind(15)
  real(rp), intent(inout) :: x
  dimension x(3)
  real(rp), parameter :: three = 3._rp
  x(1) = x(1) + x(2) + x(3) * three
  return
end subroutine

