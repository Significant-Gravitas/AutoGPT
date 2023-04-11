module mod
  integer :: i
  integer :: x(4)
  real, dimension(2,3) :: a
  real, allocatable, dimension(:,:) :: b
contains
  subroutine foo
    integer :: k
    k = 1
    a(1,2) = a(1,2)+3
  end subroutine foo
end module mod
