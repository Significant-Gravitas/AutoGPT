module foo
  public
  type, private, bind(c) :: a
     integer :: i
  end type a
  type, bind(c) :: b_
     integer :: j
  end type b_
  public :: b_
  type :: c
     integer :: k
  end type c
end module foo
