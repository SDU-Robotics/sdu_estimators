! A simple subroutine in Fortran
subroutine greet_fortran(name) bind(C, name="greet_fortran")
use, intrinsic :: iso_c_binding
character(kind=c_char), intent(in) :: name(*)
print *, "Hello from Fortran, ", trim(name)
end subroutine greet_fortran