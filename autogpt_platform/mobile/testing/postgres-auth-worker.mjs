import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { createInterface } from "node:readline";
import { once } from "node:events";
import { fileURLToPath, pathToFileURL } from "node:url";

const frontend = new URL("../../frontend/", import.meta.url);
const frontendRequire = createRequire(new URL("package.json", frontend));

export async function loadAuthRuntime() {
  const typescript = frontendRequire("typescript");
  const resolve = (specifier) =>
    specifier.startsWith("node:")
      ? specifier
      : pathToFileURL(frontendRequire.resolve(specifier)).href;
  const allowedSources = new Map([
    ["./mobile-auth-helpers", "src/lib/auth/mobile-auth-helpers.ts"],
    ["./mobile-auth", "src/lib/auth/mobile-auth.ts"],
  ]);
  const compiled = new Map();
  function compile(specifier) {
    if (compiled.has(specifier)) return compiled.get(specifier);
    const sourcePath = allowedSources.get(specifier);
    if (!sourcePath)
      throw new Error(`Unexpected frontend source import: ${specifier}`);
    let source = typescript.transpileModule(
      readFileSync(new URL(sourcePath, frontend), "utf8"),
      {
        compilerOptions: {
          module: typescript.ModuleKind.ESNext,
          target: typescript.ScriptTarget.ES2022,
        },
      },
    ).outputText;
    source = source.replace(/from\s+["']([^"']+)["']/g, (_, dependency) => {
      const target = dependency.startsWith(".")
        ? compile(dependency)
        : resolve(dependency);
      return `from ${JSON.stringify(target)}`;
    });
    const moduleURL = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
    compiled.set(specifier, moduleURL);
    return moduleURL;
  }
  const [{ betterAuth }, { admin }, { mobileAuth }, { getMigrations }, pg] =
    await Promise.all([
      import(resolve("better-auth")),
      import(resolve("better-auth/plugins")),
      import(compile("./mobile-auth")),
      import(resolve("better-auth/db/migration")),
      import(resolve("pg")),
    ]);
  function createAuth(connection, sessionBefore) {
    const pool = new (pg.Pool ?? pg.default.Pool)({
      ...connection,
      options: "-c search_path=platform",
      max: 6,
    });
    const options = {
      baseURL: "https://platform.agpt.co",
      secret: connection.authSecret,
      database: pool,
      telemetry: { enabled: false },
      logger: { disabled: true },
      rateLimit: { enabled: false },
      advanced: { database: { generateId: () => crypto.randomUUID() } },
      user: {
        modelName: "UserAuthIdentity",
        additionalFields: {
          preferredName: { type: "string", required: false },
        },
      },
      session: {
        modelName: "UserAuthSession",
        expiresIn: 60 * 60 * 24 * 30,
        cookieCache: { enabled: true, maxAge: 300 },
      },
      account: { modelName: "UserAuthAccount" },
      verification: { modelName: "UserAuthVerification" },
      emailAndPassword: { enabled: true },
      plugins: [admin(), mobileAuth()],
      ...(sessionBefore
        ? { databaseHooks: { session: { create: { before: sessionBefore } } } }
        : {}),
    };
    return { auth: betterAuth(options), options, pool };
  }
  return { createAuth, getMigrations };
}

export function cookieHeader(response) {
  return response.headers
    .getSetCookie()
    .map((cookie) => cookie.split(";")[0])
    .join("; ");
}

export async function post(auth, endpoint, body, cookie = "") {
  return auth.handler(
    new Request(`https://platform.agpt.co/api/auth/mobile/${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Origin: "https://platform.agpt.co",
        ...(cookie ? { Cookie: cookie } : {}),
      },
      body: JSON.stringify(body),
    }),
  );
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const input = createInterface({ input: process.stdin, terminal: false });
  let pool;
  try {
    const [configuration] = await once(input, "line");
    const { connection, code, verifier } = JSON.parse(configuration);
    const runtime = await loadAuthRuntime();
    const instance = runtime.createAuth(connection);
    pool = instance.pool;
    await instance.auth.$context;
    const start = once(input, "line");
    console.log(JSON.stringify({ ready: true }));
    await start;
    const response = await post(instance.auth, "exchange", {
      code,
      code_verifier: verifier,
    });
    const setCookies = response.headers.getSetCookie();
    const session =
      response.status === 200
        ? await instance.auth.api.getSession({
            headers: new Headers({ Cookie: cookieHeader(response) }),
          })
        : null;
    console.log(
      JSON.stringify({
        status: response.status,
        cookieCount: setCookies.length,
        userId: session?.user.id ?? null,
      }),
    );
  } catch (error) {
    console.log(
      JSON.stringify({
        error: error.message.replace(
          /data:text\/javascript;base64,[A-Za-z0-9+/=]+/g,
          "[compiled frontend module]",
        ),
      }),
    );
    process.exitCode = 1;
  } finally {
    input.close();
    if (pool) await pool.end();
  }
}
