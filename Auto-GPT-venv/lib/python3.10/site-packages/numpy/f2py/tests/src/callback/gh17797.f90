function gh17797(f, y) result(r)
  external f
  integer(8) :: r, f
  integer(8), dimension(:) :: y
  r = f(0)
  r = r + sum(y)
end function gh17797
