module fortfuncs
  implicit none
contains
  subroutine square(x,y)
    integer, intent(in), value :: x
    integer, intent(out) :: y
    y = x*x
  end subroutine square
end module fortfuncs
