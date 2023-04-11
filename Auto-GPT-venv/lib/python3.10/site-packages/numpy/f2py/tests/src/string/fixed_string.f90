function sint(s) result(i)
   implicit none
   character(len=*) :: s
   integer :: j, i
   i = 0
   do j=len(s), 1, -1
    if (.not.((i.eq.0).and.(s(j:j).eq.' '))) then
      i = i + ichar(s(j:j)) * 10 ** (j - 1)
    endif
   end do
   return
 end function sint

 function test_in_bytes4(a) result (i)
   implicit none
   integer :: sint
   character(len=4) :: a
   integer :: i
   i = sint(a)
   a(1:1) = 'A'
   return
 end function test_in_bytes4

 function test_inout_bytes4(a) result (i)
   implicit none
   integer :: sint
   character(len=4), intent(inout) :: a
   integer :: i
   if (a(1:1).ne.' ') then
     a(1:1) = 'E'
   endif
   i = sint(a)
   return
 end function test_inout_bytes4
