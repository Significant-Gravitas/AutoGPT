subroutine foo(is_, ie_, arr, tout)
 implicit none
 integer :: is_,ie_
 real, intent(in) :: arr(is_:ie_)
 real, intent(out) :: tout(is_:ie_)
 tout = arr
end
