! Check that parameters are correct intercepted.
! Specifically that types of constants without 
! compound kind specs are correctly inferred
! adapted Gibbs iteration code from pymc 
! for this test case 
subroutine foo_non_compound_int(x)
  implicit none
  integer, parameter :: ii = selected_int_kind(9)

  integer(ii)   maxiterates
  parameter (maxiterates=2)

  integer(ii)   maxseries
  parameter (maxseries=2)

  integer(ii)   wasize
  parameter (wasize=maxiterates*maxseries)
  integer(ii), intent(inout) :: x
  dimension x(wasize)

  x(1) = x(1) + x(2) + x(3) + x(4) * wasize
  return
end subroutine
