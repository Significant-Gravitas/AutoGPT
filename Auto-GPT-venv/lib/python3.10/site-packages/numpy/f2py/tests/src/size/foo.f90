
subroutine foo(a, n, m, b)
  implicit none

  real, intent(in) :: a(n, m)
  integer, intent(in) :: n, m
  real, intent(out) :: b(size(a, 1))

  integer :: i

  do i = 1, size(b)
    b(i) = sum(a(i,:))
  enddo
end subroutine

subroutine trans(x,y)
  implicit none
  real, intent(in), dimension(:,:) :: x
  real, intent(out), dimension( size(x,2), size(x,1) ) :: y
  integer :: N, M, i, j
  N = size(x,1)
  M = size(x,2)
  DO i=1,N
     do j=1,M
        y(j,i) = x(i,j)
     END DO
  END DO
end subroutine trans

subroutine flatten(x,y)
  implicit none
  real, intent(in), dimension(:,:) :: x
  real, intent(out), dimension( size(x) ) :: y
  integer :: N, M, i, j, k
  N = size(x,1)
  M = size(x,2)
  k = 1
  DO i=1,N
     do j=1,M
        y(k) = x(i,j)
        k = k + 1
     END DO
  END DO
end subroutine flatten
