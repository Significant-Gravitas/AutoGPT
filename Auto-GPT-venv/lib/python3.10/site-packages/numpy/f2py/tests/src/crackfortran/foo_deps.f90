module foo
  type bar
    character(len = 4) :: text
  end type bar
  type(bar), parameter :: abar = bar('abar')
end module foo
