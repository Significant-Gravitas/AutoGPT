! Check that parameters are correct intercepted.
! Constants with comma separations are commonly
! used, for instance Pi = 3._dp
subroutine foo(x)
  implicit none
  integer, parameter :: sp = selected_real_kind(6)
  integer, parameter :: dp = selected_real_kind(15)
  integer, parameter :: ii = selected_int_kind(9)
  integer, parameter :: il = selected_int_kind(18)
  real(dp), intent(inout) :: x
  dimension x(3)
  real(sp), parameter :: three_s = 3._sp
  real(dp), parameter :: three_d = 3._dp
  integer(ii), parameter :: three_i = 3_ii
  integer(il), parameter :: three_l = 3_il
  x(1) = x(1) + x(2) * three_s * three_i + x(3) * three_d * three_l
  x(2) = x(2) * three_s
  x(3) = x(3) * three_l
  return
end subroutine


subroutine foo_no(x)
  implicit none
  integer, parameter :: sp = selected_real_kind(6)
  integer, parameter :: dp = selected_real_kind(15)
  integer, parameter :: ii = selected_int_kind(9)
  integer, parameter :: il = selected_int_kind(18)
  real(dp), intent(inout) :: x
  dimension x(3)
  real(sp), parameter :: three_s = 3.
  real(dp), parameter :: three_d = 3.
  integer(ii), parameter :: three_i = 3
  integer(il), parameter :: three_l = 3
  x(1) = x(1) + x(2) * three_s * three_i + x(3) * three_d * three_l
  x(2) = x(2) * three_s
  x(3) = x(3) * three_l
  return
end subroutine

subroutine foo_sum(x)
  implicit none
  integer, parameter :: sp = selected_real_kind(6)
  integer, parameter :: dp = selected_real_kind(15)
  integer, parameter :: ii = selected_int_kind(9)
  integer, parameter :: il = selected_int_kind(18)
  real(dp), intent(inout) :: x
  dimension x(3)
  real(sp), parameter :: three_s = 2._sp + 1._sp
  real(dp), parameter :: three_d = 1._dp + 2._dp
  integer(ii), parameter :: three_i = 2_ii + 1_ii
  integer(il), parameter :: three_l = 1_il + 2_il
  x(1) = x(1) + x(2) * three_s * three_i + x(3) * three_d * three_l
  x(2) = x(2) * three_s
  x(3) = x(3) * three_l
  return
end subroutine
