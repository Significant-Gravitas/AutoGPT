        ! When gh18335_workaround is defined as an extension,
        ! the issue cannot be reproduced.
        !subroutine gh18335_workaround(f, y)
        !  implicit none
        !  external f
        !  integer(kind=1) :: y(1)
        !  call f(y)
        !end subroutine gh18335_workaround

        function gh18335(f) result (r)
          implicit none
          external f
          integer(kind=1) :: y(1), r
          y(1) = 123
          call f(y)
          r = y(1)
        end function gh18335
