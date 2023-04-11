

subroutine selectedrealkind(p, r, res)
  implicit none
  
  integer, intent(in) :: p, r
  !f2py integer :: r=0
  integer, intent(out) :: res
  res = selected_real_kind(p, r)

end subroutine

subroutine selectedintkind(p, res)
  implicit none

  integer, intent(in) :: p
  integer, intent(out) :: res
  res = selected_int_kind(p)

end subroutine
