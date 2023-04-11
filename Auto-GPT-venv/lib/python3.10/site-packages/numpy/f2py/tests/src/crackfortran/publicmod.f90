module foo
  public
  integer, private :: a
  public :: setA
contains
  subroutine setA(v)
    integer, intent(in) :: v
    a = v
  end subroutine setA
end module foo
