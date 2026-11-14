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

  // The executor binds stored credentials to both URLs, so `start` must name
  // the same deployment `auth` signed in to, or it finds no credentials.
  const deploymentArguments = `--platform-url ${platformArgument} --platform-oauth-url ${oauthArgument}`;

  return {
    platformURL,
    authCommand: `autogpt-shim ${deploymentArguments} auth`,
    startCommand: `autogpt-shim ${deploymentArguments} start`,
    config: `platform_url = ${JSON.stringify(platformURL)}\nplatform_oauth_url = ${JSON.stringify(oauthURL)}`,
  };
}
