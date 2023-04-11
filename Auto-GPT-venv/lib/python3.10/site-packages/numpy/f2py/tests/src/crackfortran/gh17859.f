        integer(8) function external_as_statement(fcn)
        implicit none
        external fcn
        integer(8) :: fcn
        external_as_statement = fcn(0)
        end

        integer(8) function external_as_attribute(fcn)
        implicit none
        integer(8), external :: fcn
        external_as_attribute = fcn(0)
        end
