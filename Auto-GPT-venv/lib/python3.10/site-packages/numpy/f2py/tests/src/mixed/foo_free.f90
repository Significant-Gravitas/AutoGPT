module foo_free
contains
  subroutine bar13(a)
    !f2py intent(out) a
    integer a
    a = 13
  end subroutine bar13
end module foo_free
