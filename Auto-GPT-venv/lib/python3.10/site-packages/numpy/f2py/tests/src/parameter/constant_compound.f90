! Check that parameters are correct intercepted.
! Constants with comma separations are commonly
! used, for instance Pi = 3._dp
subroutine foo_compound_int(x)
  implicit none
  integer, parameter :: ii = selected_int_kind(9)
  integer(ii), intent(inout) :: x
  dimension x(3)
  integer(ii), parameter :: three = 3_ii
  integer(ii), parameter :: two = 2_ii
  integer(ii), parameter :: six = three * 1_ii * two

  x(1) = x(1) + x(2) + x(3) * six
  return
end subroutine
