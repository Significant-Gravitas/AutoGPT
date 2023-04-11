! Check that parameters are correct intercepted.
! Constants with comma separations are commonly
! used, for instance Pi = 3._dp
subroutine foo_int(x)
  implicit none
  integer, parameter :: ii = selected_int_kind(9)
  integer(ii), intent(inout) :: x
  dimension x(3)
  integer(ii), parameter :: three = 3_ii
  x(1) = x(1) + x(2) + x(3) * three
  return
end subroutine

subroutine foo_long(x)
  implicit none
  integer, parameter :: ii = selected_int_kind(18)
  integer(ii), intent(inout) :: x
  dimension x(3)
  integer(ii), parameter :: three = 3_ii
  x(1) = x(1) + x(2) + x(3) * three
  return
end subroutine
