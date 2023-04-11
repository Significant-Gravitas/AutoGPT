MODULE char_test

CONTAINS

SUBROUTINE change_strings(strings, n_strs, out_strings)
    IMPLICIT NONE

    ! Inputs
    INTEGER, INTENT(IN) :: n_strs
    CHARACTER, INTENT(IN), DIMENSION(2,n_strs) :: strings
    CHARACTER, INTENT(OUT), DIMENSION(2,n_strs) :: out_strings

!f2py INTEGER, INTENT(IN) :: n_strs
!f2py CHARACTER, INTENT(IN), DIMENSION(2,n_strs) :: strings
!f2py CHARACTER, INTENT(OUT), DIMENSION(2,n_strs) :: strings

    ! Misc.
    INTEGER*4 :: j


    DO j=1, n_strs
        out_strings(1,j) = strings(1,j)
        out_strings(2,j) = 'A'
    END DO

END SUBROUTINE change_strings

END MODULE char_test

