module foo
  private
  integer :: a
  public :: setA
  integer :: b
contains
  subroutine setA(v)
    integer, intent(in) :: v
    a = v
  end subroutine setA
end module foo
