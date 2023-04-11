      SUBROUTINE INITCB
      DOUBLE PRECISION LONG
      CHARACTER        STRING
      INTEGER          OK
    
      COMMON  /BLOCK/ LONG, STRING, OK
      LONG = 1.0
      STRING = '2'
      OK = 3
      RETURN
      END
