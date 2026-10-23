# Keyless API Access

Call the AutoGPT External API from a CI job, a server or a website without an API key.

An API key is a long-lived secret. It gets pasted into CI settings, environment variables, logs and chats, and it works for anyone who has a copy until someone revokes it. Keyless access uses something your code already has instead: the identity its platform gives it. GitHub Actions, Vercel, GitLab, Google Cloud and others can hand your code a short-lived, signed token that says exactly which repository, project or service account is running. AutoGPT checks that token and gives back its own token that expires within the hour.

|  | API key | Keyless |
|---|---|---|
| What your code stores | A secret that never expires | Nothing (or, on your own server, a private key that never leaves it) |
| If it leaks | Works from anywhere until revoked | The AutoGPT token expires in minutes and only your workload can get a new one |
| Rotation | Create a new key, update every place that uses it | Nothing to rotate |
| How you call the API | `X-API-Key: agpt_…` | `Authorization: Bearer agpt_wt_…` |

## How it works

1. **Your platform vouches for your code.** It signs a short token, an OpenID Connect (OIDC) ID token, saying who is running: for example "repository `acme/api`, branch `main`". The platform publishes the public half of its signing key, so anyone can check the signature and nobody can forge one.
2. **You tell AutoGPT which workload to trust.** In **Settings → Keyless Access**, you name the platform and the exact repository, project, service account or server, and choose what it may do.
3. **Your code swaps the platform's token for an AutoGPT token.** AutoGPT checks the signature against the platform's public key and confirms the token was made for AutoGPT. It also confirms the token names the workload you trusted, then returns a token that lasts 15 minutes.
4. **Your code calls the API with it,** exactly as it would with an API key, but in the `Authorization` header. When the token expires, it does step 3 again.

## Set it up

1. Open **Settings → Keyless Access** and click **Add workload**.
2. Pick where your code runs, fill in the repository, project, service account or site, and choose the permissions it needs (the same [scopes](api-guide.md#available-scopes) as API keys).
3. The page then shows the commands for that workload with its ID filled in. You can see them again any time with **How to use**.

The workload ID is not a secret. It only tells AutoGPT which settings to check the token against, so it is safe to put in a workflow file or an environment variable.

## Use it with curl

Every platform follows the same three steps. Only the first one differs, and the platform guides below cover it.

```bash
# 1. Get an identity token from your platform (see the platform guides below)
OIDC_TOKEN="..."

# 2. Swap it for an AutoGPT token
WORKLOAD_ID="<your workload ID>"

AUTOGPT_TOKEN=$(curl -s https://backend.agpt.co/api/oauth/token \
  -H "Content-Type: application/json" \
  -d "{\"grant_type\": \"urn:ietf:params:oauth:grant-type:token-exchange\",
       \"subject_token_type\": \"urn:ietf:params:oauth:token-type:id_token\",
       \"client_id\": \"$WORKLOAD_ID\",
       \"subject_token\": \"$OIDC_TOKEN\"}" \
  | jq -r .access_token)

# 3. Call the API. With an API key this would be -H "X-API-Key: $AUTOGPT_API_KEY"
curl https://backend.agpt.co/external-api/v1/me \
  -H "Authorization: Bearer $AUTOGPT_TOKEN"
```

Step 2 also accepts an optional `scope` (space-separated permissions) to ask for fewer permissions than the workload has. The response looks like this:

```json
{
  "access_token": "agpt_wt_…",
  "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
  "token_type": "Bearer",
  "expires_in": 900,
  "access_token_expires_at": "2026-10-23T12:15:00Z",
  "scope": "IDENTITY EXECUTE_GRAPH"
}
```

{% hint style="info" %}
**Self-hosting?** Use your own server's address instead of `https://backend.agpt.co`, both in the commands and as the token audience.
{% endhint %}

## Platform guides

In every guide, the audience is the address AutoGPT checks tokens against, `https://backend.agpt.co` on the hosted platform. The **How to use** view shows the exact value for your workload.

### GitHub Actions

Trust a repository, and optionally one branch or one GitHub environment. The job needs permission to request an identity token:

```yaml
permissions:
  id-token: write
  contents: read

jobs:
  call-autogpt:
    runs-on: ubuntu-latest
    steps:
      - name: Call AutoGPT
        env:
          WORKLOAD_ID: ${{ vars.AUTOGPT_WORKLOAD_ID }}
        run: |
          OIDC_TOKEN=$(curl -s \
            -H "Authorization: Bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
            "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=https://backend.agpt.co" | jq -r .value)
          # then steps 2 and 3 from above
```

### GitLab CI

Trust a gitlab.com project, and optionally one branch. For a self-managed GitLab, choose **Another OIDC provider** with your instance's URL as the issuer.

```yaml
call-autogpt:
  id_tokens:
    AUTOGPT_ID_TOKEN:
      aud: https://backend.agpt.co
  script:
    - OIDC_TOKEN="$AUTOGPT_ID_TOKEN"
    # then steps 2 and 3 from above
```

### Vercel

Trust one project of your team, in one environment or all of them. Check your project's issuer mode under **Project Settings → Security → Secure backend access**; new projects use the team issuer. In a Vercel Function:

```js
import { getVercelOidcToken } from "@vercel/oidc";

const oidcToken = await getVercelOidcToken({ audience: "https://backend.agpt.co" });

const exchange = await fetch("https://backend.agpt.co/api/oauth/token", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    grant_type: "urn:ietf:params:oauth:grant-type:token-exchange",
    subject_token_type: "urn:ietf:params:oauth:token-type:id_token",
    client_id: process.env.AUTOGPT_WORKLOAD_ID,
    subject_token: oidcToken,
  }),
});
const { access_token } = await exchange.json();

const me = await fetch("https://backend.agpt.co/external-api/v1/me", {
  headers: { Authorization: `Bearer ${access_token}` },
});
```

### Google Cloud

Trust a service account. Code running as it on Compute Engine, Cloud Run or GKE (with Workload Identity) asks the metadata server for a token. `format=full` adds the email claim AutoGPT checks.

```bash
OIDC_TOKEN=$(curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=https://backend.agpt.co&format=full")
```

### Your own website or server

A server that isn't on one of those platforms can still go keyless: your site becomes the issuer. It publishes a public key on its own domain, and your server signs tokens with the matching private key. AutoGPT reads the public key from your site, so only code that holds your private key can get an AutoGPT token for this workload.

The private key never leaves your server and is never sent anywhere, unlike an API key, which travels with every request. To rotate it, publish a new public key on your site; nothing changes on AutoGPT.

1. **Make a key pair.** In Node.js (`npm install jose`):

   ```js
   import { exportJWK, exportPKCS8, generateKeyPair } from "jose";

   const { publicKey, privateKey } = await generateKeyPair("ES256", { extractable: true });
   const jwk = await exportJWK(publicKey);

   console.log(await exportPKCS8(privateKey)); // keep on your server only
   console.log(JSON.stringify({ keys: [{ ...jwk, kid: "autogpt-1", alg: "ES256", use: "sig" }] }));
   ```

   Or with OpenSSL and Python (`pip install "pyjwt[crypto]"`):

   ```bash
   openssl ecparam -name prime256v1 -genkey -noout | openssl pkcs8 -topk8 -nocrypt -out autogpt-signing-key.pem
   ```

   ```python
   import json
   import jwt
   from cryptography.hazmat.primitives import serialization

   with open("autogpt-signing-key.pem", "rb") as f:
       key = serialization.load_pem_private_key(f.read(), password=None)
   jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(key.public_key()))
   print(json.dumps({"keys": [{**jwk, "kid": "autogpt-1", "alg": "ES256", "use": "sig"}]}))
   ```

2. **Publish two files on your site.** `https://acme.com/.well-known/openid-configuration`:

   ```json
   {
     "issuer": "https://acme.com",
     "jwks_uri": "https://acme.com/.well-known/jwks.json"
   }
   ```

   and the key set printed in step 1 at `https://acme.com/.well-known/jwks.json`. The `issuer` must match the site you entered in AutoGPT exactly, with no trailing slash.

3. **Sign a short token whenever you need an AutoGPT token.** Its subject is the server name you entered in AutoGPT:

   ```js
   import { importPKCS8, SignJWT } from "jose";

   const key = await importPKCS8(process.env.AUTOGPT_SIGNING_KEY, "ES256");
   const oidcToken = await new SignJWT({})
     .setProtectedHeader({ alg: "ES256", kid: "autogpt-1" })
     .setIssuer("https://acme.com")
     .setSubject("backend")
     .setAudience("https://backend.agpt.co")
     .setIssuedAt()
     .setExpirationTime("5m")
     .sign(key);
   ```

   ```python
   import time
   import jwt

   now = int(time.time())
   oidc_token = jwt.encode(
       {"iss": "https://acme.com", "sub": "backend", "aud": "https://backend.agpt.co", "iat": now, "exp": now + 300},
       open("autogpt-signing-key.pem").read(),
       algorithm="ES256",
       headers={"kid": "autogpt-1"},
   )
   ```

Then do steps 2 and 3 from [Use it with curl](#use-it-with-curl).

### Other OIDC providers

Kubernetes service accounts, Auth0, self-managed GitLab and any other provider that publishes an OpenID Connect discovery document work too. Enter its issuer URL, and a claim that names your workload with the value it must have. For example, Kubernetes uses `sub` = `system:serviceaccount:<namespace>:<name>`. AutoGPT refuses settings that would trust every user of a shared provider.

## Calling from a browser

Don't call the External API from code that runs in visitors' browsers, with any kind of credential. Anyone can read browser code and copy what it sends. Make the call from your server, using one of the setups above, and have your page talk to your server. If you want each visitor to act as their own AutoGPT account, use [Sign in with AutoGPT](oauth-guide.md) instead.

## Managing workloads

**Settings → Keyless Access** lists every trusted workload with what it trusts and when it last got a token.

- **Disable** pauses a workload: it can't get new tokens, and tokens it already has stop working until you enable it again.
- **Revoke** removes it for good, along with every token it holds.

To change what a workload trusts, add a new one and revoke the old one.

## Troubleshooting

The token exchange answers `400` with an `error` and an `error_description`:

| `error_description` | What to check |
|---|---|
| `Unknown or inactive workload identity` | The workload ID is wrong, or the workload is disabled or revoked. |
| `Token rejected: Audience doesn't match` | Request the platform token with exactly the audience shown in **How to use**. |
| `Token rejected: Invalid issuer` | The token comes from a different issuer than the one you trusted, for example the Vercel team issuer instead of the global one. |
| `Token rejected: Signature has expired` | Get a fresh platform token; they last minutes to hours. |
| `Token does not meet the workload identity's claim conditions` | The token is for a different repository, branch, project, environment or server name than the one you trusted. |
| `Could not fetch the issuer's signing keys` | For your own site: both `/.well-known` files must be reachable over https, and the discovery document's `issuer` must match exactly. |
| `Token was not signed by a current key of its issuer` | The `kid` in your token's header must match a key in your published key set. |
| `Not granted to this workload identity: …` (`invalid_scope`) | You asked for a permission the workload doesn't have. |

A call to the External API with an expired AutoGPT token answers `401 Workload token expired`. Do step 2 again.
