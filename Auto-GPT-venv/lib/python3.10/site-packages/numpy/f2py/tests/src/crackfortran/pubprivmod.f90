module foo
  public
  integer, private :: a
  integer :: b
contains
  subroutine setA(v)
    integer, intent(in) :: v
    a = v
  end subroutine setA
end module foo
