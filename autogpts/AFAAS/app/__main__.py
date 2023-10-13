import uvicorn
import app.sdk.forge_log
LOG = app.sdk.forge_log.ForgeLogger(__name__)


logo = """\n\n
    A      FFFFF   A     A       SSSSSS 
   A A     F      A A   A A     S     
  A   A    FFFF  A   A A   A    SSSSS  
 AAAAAAA   F    AAAAAAAAAAAAA        S 
A       A  F   A     AA      ASSSSSSSS 
\n\n"""



if __name__ == "__main__":
    from api import api , port 
    print(logo)
    uvicorn.run(
        "api:api", host="localhost", port=port, log_level="error", reload=True
    )
