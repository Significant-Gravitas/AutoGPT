module foo
  type bar
     character(len = 32) :: item
  end type bar
  interface operator(.item.)
     module procedure item_int, item_real
  end interface operator(.item.)
  interface operator(==)
     module procedure items_are_equal
  end interface operator(==)
  interface assignment(=)
     module procedure get_int, get_real
  end interface assignment(=)
contains
  function item_int(val) result(elem)
    integer, intent(in) :: val
    type(bar) :: elem

    write(elem%item, "(I32)") val
  end function item_int

  function item_real(val) result(elem)
    real, intent(in) :: val
    type(bar) :: elem

    write(elem%item, "(1PE32.12)") val
  end function item_real

  function items_are_equal(val1, val2) result(equal)
    type(bar), intent(in) :: val1, val2
    logical :: equal

    equal = (val1%item == val2%item)
  end function items_are_equal

  subroutine get_real(rval, item)
    real, intent(out) :: rval
    type(bar), intent(in) :: item

    read(item%item, *) rval
  end subroutine get_real

  subroutine get_int(rval, item)
    integer, intent(out) :: rval
    type(bar), intent(in) :: item

    read(item%item, *) rval
  end subroutine get_int
end module foo
