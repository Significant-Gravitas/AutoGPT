export function localExecutorDeployment(
  apiURL: string,
  frontendOrigin: string,
) {
  const backend = new URL(apiURL, frontendOrigin);
  backend.pathname = backend.pathname
    .replace(/\/api\/?$/, "")
    .replace(/\/$/, "");
  backend.search = "";
  backend.hash = "";
  const platformURL = backend.toString().replace(/\/$/, "");
  const oauthURL = new URL("/auth", frontendOrigin).toString();
  const platformArgument = `'${platformURL.replaceAll("'", "%27")}'`;
  const oauthArgument = `'${oauthURL.replaceAll("'", "%27")}'`;

  return {
    platformURL,
    authCommand: `autogpt-shim --platform-url ${platformArgument} --platform-oauth-url ${oauthArgument} auth`,
    startCommand: `autogpt-shim --platform-url ${platformArgument} start`,
    config: `platform_url = ${JSON.stringify(platformURL)}\nplatform_oauth_url = ${JSON.stringify(oauthURL)}`,
  };
}
