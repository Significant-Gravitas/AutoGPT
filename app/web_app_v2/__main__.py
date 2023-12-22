import uvicorn

from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


logo = """\n\n
          AAA         FFFFFFFFFFFF      AAA              AAA           SSSSSSSSSSSSSSSSSSSSSSS 
         AAAAA        FFFFFFFFFF       AAAAA            AAAAA        SSSSSSSSSSSSSSSSSSSSSSSS     
        AA   AA       FF              AA   AA          AA   AA      SSS   
       AA     AA      FF             AA     AA        AA     AA      SSSS 
      AA       AA     FF            AA       AA      AA       AA      SSSSSSSSSSSSSSS 
     AAAAAAAAAAAAA    FFFFFF       AAAAAAAAAAAAA    AAAAAAAAAAAAA         SSSSSSSSSSSSS
    AAAAAAAAAAAAAAA   FF          AAAAAAAAAAAAAAA  AAAAAAAAAAAAAAA                  SSSS
   AA             AA  FF         AA             AAAA             AA                 SSSS
  AA               AA FF        AA               AAA              AA  SSSSSSSSSSSSSSSSS
 AA                 AAFF       AA                AAA               AASSSSSSSSSSSSSSSS
  \n\n"""


if __name__ == "__main__":
    from api import api, port

    print(logo)
    uvicorn.run("api:api", host="localhost", port=port, log_level="error", reload=True)
