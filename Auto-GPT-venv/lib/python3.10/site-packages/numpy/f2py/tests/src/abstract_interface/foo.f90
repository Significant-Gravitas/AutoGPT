module ops_module

  abstract interface
    subroutine op(x, y, z)
      integer, intent(in) :: x, y
      integer, intent(out) :: z
    end subroutine
  end interface

contains

  subroutine foo(x, y, r1, r2)
    integer, intent(in) :: x, y
    integer, intent(out) :: r1, r2
    procedure (op) add1, add2
    procedure (op), pointer::p
    p=>add1
    call p(x, y, r1)
    p=>add2
    call p(x, y, r2)
  end subroutine
end module

subroutine add1(x, y, z)
  integer, intent(in) :: x, y
  integer, intent(out) :: z
  z = x + y
end subroutine

subroutine add2(x, y, z)
  integer, intent(in) :: x, y
  integer, intent(out) :: z
  z = x + 2 * y
end subroutine
