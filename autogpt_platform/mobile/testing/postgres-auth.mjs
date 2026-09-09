import assert from "node:assert/strict";
import { createHash, randomBytes } from "node:crypto";
import { execFile, spawn } from "node:child_process";
import { once } from "node:events";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { promisify, parseArgs } from "node:util";
import {
  cookieHeader,
  loadAuthRuntime,
  post,
} from "./postgres-auth-worker.mjs";

const execute = promisify(execFile);
const { values } = parseArgs({
  options: { image: { type: "string", default: "postgres:17" } },
});
const containerName = `autogpt-native-auth-${randomBytes(6).toString("hex")}`;
const password = randomBytes(32).toString("base64url");
const workers = [];
let containerID;
let pool;
let cleaning;

async function docker(args, options = {}) {
  return (
    await execute("docker", args, {
      timeout: 30_000,
      maxBuffer: 1024 * 1024,
      ...options,
    })
  ).stdout.trim();
}

async function cleanup() {
  if (cleaning) return cleaning;
  cleaning = (async () => {
    for (const worker of workers)
      if (worker.exitCode === null) worker.kill("SIGTERM");
    if (pool) await pool.end();
    if (containerID) {
      await docker(["rm", "--force", containerID]);
      console.log(
        "Removed the disposable PostgreSQL container and its tmpfs data.",
      );
    }
  })();
  return cleaning;
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, async () => {
    try {
      await cleanup();
    } finally {
      process.exit(signal === "SIGINT" ? 130 : 143);
    }
  });
}

function handoff() {
  const verifier = randomBytes(32).toString("base64url");
  return {
    verifier,
    challenge: createHash("sha256").update(verifier).digest("base64url"),
    state: randomBytes(32).toString("base64url"),
  };
}

async function createUser(auth, label) {
  const response = await auth.api.signUpEmail({
    body: {
      name: "Native PostgreSQL fixture",
      email: `${label}@example.invalid`,
      password: randomBytes(24).toString("base64url"),
    },
    asResponse: true,
  });
  assert.equal(response.status, 200, `Could not create ${label} fixture user`);
  return {
    id: (await response.json()).user.id,
    cookie: cookieHeader(response),
  };
}

async function issueCode(auth, user, request = handoff()) {
  const response = await post(
    auth,
    "authorize",
    {
      code_challenge: request.challenge,
      state: request.state,
      expected_user_id: user.id,
    },
    user.cookie,
  );
  assert.equal(response.status, 200, "Could not issue fixture handoff");
  return {
    ...request,
    code: new URL((await response.json()).url).searchParams.get("code"),
  };
}

async function prepareWorker(connection, request) {
  const worker = spawn(
    process.execPath,
    [fileURLToPath(new URL("postgres-auth-worker.mjs", import.meta.url))],
    { stdio: ["pipe", "pipe", "pipe"] },
  );
  workers.push(worker);
  let stderr = "";
  worker.stderr.on("data", (data) => {
    stderr += data;
  });
  const output = createInterface({ input: worker.stdout, terminal: false });
  const messages = [];
  const pending = [];
  output.on("line", (line) => {
    const message = JSON.parse(line);
    if (pending.length) pending.shift()(message);
    else messages.push(message);
  });
  function nextMessage() {
    if (messages.length) return Promise.resolve(messages.shift());
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error("Exchange worker timed out")),
        30_000,
      );
      pending.push((message) => {
        clearTimeout(timeout);
        resolve(message);
      });
    });
  }
  const exited = once(worker, "exit");
  worker.stdin.write(
    `${JSON.stringify({ connection, code: request.code, verifier: request.verifier })}\n`,
  );
  assert.deepEqual(await nextMessage(), { ready: true });
  return {
    async exchange() {
      const result = nextMessage();
      worker.stdin.end("exchange\n");
      const message = await result;
      const [exitCode] = await exited;
      output.close();
      assert.equal(exitCode, 0, `Exchange worker failed: ${stderr}`);
      assert.equal(message.error, undefined);
      return message;
    },
  };
}

try {
  await docker(["info", "--format", "{{.ServerVersion}}"]);
  await docker(["image", "inspect", values.image, "--format", "{{.Id}}"]);
  console.log(
    `Using cached ${values.image}; no image download or existing database access.`,
  );
  containerID = await docker(
    [
      "run",
      "--detach",
      "--rm",
      "--pull",
      "never",
      "--name",
      containerName,
      "--label",
      "com.agpt.purpose=native-auth-fixture",
      "--publish",
      "127.0.0.1::5432",
      "--memory",
      "256m",
      "--cpus",
      "1",
      "--tmpfs",
      "/var/lib/postgresql/data:rw,size=256m",
      "--env",
      "POSTGRES_PASSWORD",
      "--env",
      "POSTGRES_USER=native_auth_fixture",
      "--env",
      "POSTGRES_DB=native_auth_fixture",
      values.image,
      "postgres",
      "-c",
      "shared_buffers=32MB",
      "-c",
      "max_connections=32",
    ],
    { env: { ...process.env, POSTGRES_PASSWORD: password } },
  );
  const binding = await docker(["port", containerID, "5432/tcp"]);
  const port = Number(/^127\.0\.0\.1:(\d+)$/.exec(binding)?.[1]);
  assert.ok(port, "Test database did not bind exclusively to loopback");
  const connection = {
    host: "127.0.0.1",
    port,
    user: "native_auth_fixture",
    password,
    database: "native_auth_fixture",
    authSecret: randomBytes(32).toString("base64url"),
  };
  const runtime = await loadAuthRuntime();
  let revokeDuringCreate = null;
  const instance = runtime.createAuth(connection, async (_, context) => {
    if (revokeDuringCreate) {
      const source = revokeDuringCreate;
      revokeDuringCreate = null;
      await context.context.internalAdapter.deleteSession(source);
    }
  });
  pool = instance.pool;
  const { auth } = instance;
  let ready = false;
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      await pool.query("SELECT 1");
      ready = true;
      break;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
  assert.ok(ready, "Disposable PostgreSQL did not become ready");
  await pool.query("CREATE SCHEMA platform");
  await (await runtime.getMigrations(instance.options)).runMigrations();
  const context = await auth.$context;
  console.log(
    "Created fresh platform.UserAuth* tables using Better Auth migrations.",
  );

  const user = await createUser(auth, "concurrent");
  const request = await issueCode(auth, user);
  const contenders = await Promise.all(
    Array.from({ length: 4 }, () => prepareWorker(connection, request)),
  );
  const results = await Promise.all(
    contenders.map((worker) => worker.exchange()),
  );
  assert.deepEqual(
    results.map((result) => result.status).sort(),
    [200, 400, 400, 400],
  );
  const winner = results.find((result) => result.status === 200);
  assert.equal(winner.userId, user.id);
  assert.ok(winner.cookieCount >= 2);
  assert.ok(
    results
      .filter((result) => result.status !== 200)
      .every((result) => result.cookieCount === 0),
  );
  assert.equal(
    (
      await post(auth, "exchange", {
        code: request.code,
        code_verifier: request.verifier,
      })
    ).status,
    400,
  );
  console.log(
    "PASS: four separate Node processes exchanged one ticket; exactly one won and replay failed.",
  );

  const cookieRequest = await issueCode(auth, user);
  const cookieResponse = await post(auth, "exchange", {
    code: cookieRequest.code,
    code_verifier: cookieRequest.verifier,
  });
  assert.equal(cookieResponse.status, 200);
  assert.deepEqual(await cookieResponse.json(), { success: true });
  const secureCookie = cookieResponse.headers
    .getSetCookie()
    .find((cookie) => cookie.startsWith("__Secure-better-auth.session_token="));
  for (const attribute of ["HttpOnly", "Secure", "SameSite=Lax"])
    assert.ok(secureCookie.includes(attribute));
  assert.ok(cookieResponse.headers.get("cache-control").includes("no-store"));
  console.log(
    "PASS: secure HttpOnly cookie output authenticates the expected user without JSON token disclosure.",
  );

  const revokedUser = await createUser(auth, "revoked");
  const revokedRequest = await issueCode(auth, revokedUser);
  await auth.api.signOut({
    headers: new Headers({ Cookie: revokedUser.cookie }),
  });
  const revokedResponse = await post(auth, "exchange", {
    code: revokedRequest.code,
    code_verifier: revokedRequest.verifier,
  });
  assert.equal(revokedResponse.status, 401);
  assert.equal(revokedResponse.headers.has("set-cookie"), false);
  console.log(
    "PASS: a revoked browser session cannot establish a mobile session.",
  );

  const racingUser = await createUser(auth, "revoked-during-create");
  const racingSource = await auth.api.getSession({
    headers: new Headers({ Cookie: racingUser.cookie }),
    query: { disableCookieCache: true },
  });
  const racingRequest = await issueCode(auth, racingUser);
  revokeDuringCreate = racingSource.session.token;
  const racingResponse = await post(auth, "exchange", {
    code: racingRequest.code,
    code_verifier: racingRequest.verifier,
  });
  assert.equal(racingResponse.status, 401);
  assert.equal(racingResponse.headers.has("set-cookie"), false);
  const remaining = await pool.query(
    'SELECT COUNT(*)::int AS count FROM "UserAuthSession" WHERE "userId" = $1',
    [racingUser.id],
  );
  assert.equal(remaining.rows[0].count, 0);
  console.log(
    "PASS: revocation during app-session creation rolls back the new session before cookies are returned.",
  );

  const otherUser = await createUser(auth, "other-account");
  const staleConsent = handoff();
  const changedAccount = await post(
    auth,
    "authorize",
    {
      code_challenge: staleConsent.challenge,
      state: staleConsent.state,
      expected_user_id: user.id,
    },
    otherUser.cookie,
  );
  assert.equal(changedAccount.status, 403);
  console.log(
    "PASS: displayed consent identity cannot authorize a different current account.",
  );

  const unverified = await createUser(auth, "unverified");
  const unverifiedRequest = await issueCode(auth, unverified);
  context.options.emailAndPassword.requireEmailVerification = true;
  const deniedConsent = await post(
    auth,
    "authorize",
    {
      code_challenge: unverifiedRequest.challenge,
      state: unverifiedRequest.state,
      expected_user_id: unverified.id,
    },
    unverified.cookie,
  );
  assert.equal(deniedConsent.status, 403);
  const deniedExchange = await post(auth, "exchange", {
    code: unverifiedRequest.code,
    code_verifier: unverifiedRequest.verifier,
  });
  assert.equal(deniedExchange.status, 403);
  assert.equal(deniedExchange.headers.has("set-cookie"), false);
  console.log(
    "PASS: enabling email verification denies unverified consent and outstanding handoff exchange.",
  );
  console.log(
    "PostgreSQL native-auth validation passed. This is isolated integration evidence, not production validation.",
  );
} catch (error) {
  console.error(
    `PostgreSQL native-auth validation failed: ${error.message.replace(/data:text\/javascript;base64,[A-Za-z0-9+/=]+/g, "[compiled frontend module]")}`,
  );
  process.exitCode = 1;
} finally {
  try {
    await cleanup();
  } catch (error) {
    console.error(
      `Cleanup failed for ${containerID ?? containerName}: ${error.message.replace(/data:text\/javascript;base64,[A-Za-z0-9+/=]+/g, "[compiled frontend module]")}`,
    );
    process.exitCode = 1;
  }
}
