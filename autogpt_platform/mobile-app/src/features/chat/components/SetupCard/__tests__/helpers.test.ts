import type { CredentialMeta } from "@/lib/api/integrations";

import {
  actionLabel,
  buildSelections,
  cardIntro,
  followUpMessage,
  parseSetupRequirements,
  rowStatus,
  type RowContext,
  type RowStatus,
  type SetupRequest,
} from "../helpers";

function output(missing: Record<string, unknown>, extra: Record<string, unknown> = {}) {
  return {
    type: "setup_requirements",
    message: "To continue, please connect your GitHub account.",
    session_id: "session-1",
    setup_info: {
      agent_id: "connect_github",
      agent_name: "GitHub",
      requirements: { credentials: [], inputs: [], execution_modes: [] },
      user_readiness: { has_all_credentials: false, missing_credentials: missing, ready_to_run: false },
    },
    ...extra,
  };
}

const github = {
  provider: "github",
  provider_name: "GitHub",
  type: "oauth2",
  types: ["oauth2"],
  scopes: ["repo"],
};

function setupFor(missing: Record<string, unknown>, toolName = "connect_integration", extra = {}) {
  return parseSetupRequirements(output(missing, extra), toolName) as SetupRequest;
}

function credential(extra: Partial<CredentialMeta> = {}): CredentialMeta {
  return {
    id: "cred-1",
    provider: "github",
    type: "oauth2",
    title: "octocat",
    username: null,
    scopes: ["repo"],
    host: null,
    isSystem: false,
    serviceName: null,
    serviceIcon: null,
    ...extra,
  };
}

const empty: RowContext = { pick: undefined, credentials: [], grantedIDs: null, rejection: null };

describe("parseSetupRequirements", () => {
  it("reads a card from an object or a JSON string", () => {
    const setup = setupFor({ github_credentials: github });
    expect(setup).toMatchObject({
      toolName: "connect_integration",
      serviceName: "GitHub",
      needsInputs: false,
      inputValues: null,
      rejection: null,
    });
    expect(setup.requests.map((request) => request.key)).toEqual(["github_credentials"]);
    expect(
      parseSetupRequirements(JSON.stringify(output({ github_credentials: github })), "run_block"),
    ).not.toBeNull();
  });

  it("is no card when nothing is missing or it is another result", () => {
    expect(parseSetupRequirements(output({}), "run_mcp_tool")).toBeNull();
    expect(parseSetupRequirements({ type: "execution_started" }, "run_agent")).toBeNull();
    expect(parseSetupRequirements("not json", "run_agent")).toBeNull();
  });

  it("needs the web for a required input without a value", () => {
    const inputs = [
      { name: "repo", required: true, value: "a/b" },
      { name: "branch", required: true },
      { name: "debug", required: true, advanced: true },
    ];
    const setup = parseSetupRequirements(
      output({ github_credentials: github }, {
        setup_info: { ...output({}).setup_info, requirements: { inputs }, user_readiness: { missing_credentials: { github_credentials: github } } },
      }),
      "run_block",
    );
    expect(setup).toMatchObject({ needsInputs: true, inputValues: { repo: "a/b" } });
  });

  it("keeps a card that only needs an input, with no service to connect", () => {
    const setup = parseSetupRequirements(
      output({}, {
        setup_info: { ...output({}).setup_info, requirements: { inputs: [{ name: "file", required: true }] }, user_readiness: { missing_credentials: {} } },
      }),
      "run_block",
    );
    expect(setup).toMatchObject({ requests: [], needsInputs: true });
  });

  it("only shows a workflow's inputs, so they never send it to the web", () => {
    const setup = parseSetupRequirements(
      output({ github_credentials: github }, {
        setup_info: { ...output({}).setup_info, requirements: { inputs: [{ name: "x", required: true }] }, user_readiness: { missing_credentials: { github_credentials: github } } },
      }),
      "run_agent",
    );
    expect(setup).toMatchObject({ needsInputs: false, inputValues: null });
  });
});

describe("rowStatus", () => {
  const [request] = setupFor({ github_credentials: github }).requests;

  it("asks to sign in when no account fits", () => {
    const status = rowStatus(request, empty);
    expect(status).toEqual({ kind: "connect", method: "oauth" });
    expect(actionLabel(status)).toBe("Sign in");
  });

  it("is connected by the one account that fits", () => {
    expect(rowStatus(request, { ...empty, credentials: [credential()] })).toEqual({
      kind: "connected",
      credentialID: "cred-1",
      label: "octocat",
    });
  });

  it("is connected by an account picked in this run", () => {
    expect(rowStatus(request, { ...empty, pick: { id: "new", title: "Work" } })).toMatchObject({
      kind: "connected",
      credentialID: "new",
    });
  });

  it("asks to choose between several accounts that fit", () => {
    const status = rowStatus(request, {
      ...empty,
      credentials: [credential(), credential({ id: "cred-2", title: "work" })],
    });
    expect(status.kind).toBe("choose");
    expect(actionLabel(status)).toBe("Choose account");
  });

  it("reconnects an account short of a scope, naming it for the upgrade", () => {
    const [gapped] = setupFor({
      github_credentials: { ...github, credential_gap: { kind: "missing_scopes", credential_id: "cred-1" } },
    }).requests;
    const status = rowStatus(gapped, { ...empty, credentials: [credential({ scopes: [] })] });
    expect(status).toEqual({ kind: "reconnect", credentialID: "cred-1" });
    expect(actionLabel(status)).toBe("Reconnect");
  });

  it("never counts an account the provider refused", () => {
    const status = rowStatus(request, {
      ...empty,
      credentials: [credential()],
      rejection: { provider: "github", credentialID: "cred-1" },
    });
    expect(status).toEqual({ kind: "reconnect", credentialID: "cred-1" });
  });

  it("offers Allow for an expert lacking an account the user has", () => {
    const [ask] = setupFor({
      github_credentials: {
        ...github,
        expert_grant: { expert_id: "e1", credentials: [{ id: "cred-1", title: "octocat", type: "oauth2" }] },
      },
    }).requests;
    const context = { ...empty, credentials: [credential()], grantedIDs: [] };
    const status = rowStatus(ask, context);
    expect(status).toEqual({ kind: "allow", accounts: [{ id: "cred-1", title: "octocat" }] });
    expect(actionLabel(status)).toBe("Allow");
    expect(rowStatus(ask, { ...context, grantedIDs: ["cred-1"] }).kind).toBe("connected");
  });

  it("asks to reconnect, not allow, an offered account that needs reconnecting", () => {
    const [ask] = setupFor({
      github_credentials: {
        ...github,
        credential_gap: { kind: "reconnect_required", credential_id: "cred-1" },
        expert_grant: { expert_id: "e1", credentials: [{ id: "cred-1", title: "octocat", type: "oauth2" }] },
      },
    }).requests;
    const status = rowStatus(ask, { ...empty, credentials: [credential()], grantedIDs: [] });
    expect(status).toEqual({ kind: "reconnect", credentialID: "cred-1" });
  });

  it.each([
    [["api_key"], "Add key"],
    [["user_password"], "Add login"],
    [["host_scoped"], "Add"],
    [["device_code"], null],
  ])("labels a %j row %s", (types, label) => {
    const [row] = setupFor({ k: { provider: "svc", types } }).requests;
    expect(actionLabel(rowStatus(row, empty))).toBe(label);
  });
});

describe("buildSelections", () => {
  it("keys picks by provider, by host for a host-scoped key, and skips MCP", () => {
    const setup = setupFor({
      github_credentials: github,
      http: { provider: "http", types: ["host_scoped"], discriminator_values: ["https://api.example.com/x"] },
      mcp: { provider: "mcp", types: ["oauth2"], discriminator_values: ["https://mcp.example.com"] },
    });
    const rows = setup.requests.map((request, index) => ({ request, credentialID: `c${index}` }));
    expect(buildSelections(rows)).toEqual({ github: "c0", "http:api.example.com": "c1" });
  });
});

describe("followUpMessage", () => {
  const missing = { github_credentials: github };

  it.each([
    ["connect_integration", "I've configured the required credentials. I've connected my account. Please continue."],
    ["run_block", "I've configured the required credentials. Please re-run this step now."],
    ["run_agent", "I've configured the required credentials. Please check if everything is ready and proceed with running the workflow."],
    ["run_mcp_tool", "I've connected the integration. Please retry the same call."],
  ])("sends the web's line for %s", (toolName, message) => {
    expect(followUpMessage(setupFor(missing, toolName), [])).toBe(message);
  });

  it("treats an MCP capability call as an MCP card, but not a workflow that needs MCP", () => {
    const mcp = { mcp_credentials: { provider: "mcp", types: ["oauth2"] } };
    expect(followUpMessage(setupFor(mcp, "run_capability"), [])).toBe(
      "I've connected the integration. Please retry the same call.",
    );
    expect(followUpMessage(setupFor(mcp, "run_agent"), [])).toBe(
      "I've configured the required credentials. Please check if everything is ready and proceed with running the workflow.",
    );
  });

  it("carries the picked accounts for a trigger", () => {
    const setup = setupFor(missing, "setup_agent_webhook_trigger");
    expect(followUpMessage(setup, [{ request: setup.requests[0], credentialID: "c1" }])).toBe(
      'I\'ve selected the credentials to use. Call setup_agent_webhook_trigger with credentials={"github_credentials":"c1"} to finish setting up the trigger.',
    );
  });

  it("carries a block's filled-in inputs", () => {
    const setup = { ...setupFor(missing, "run_block"), inputValues: { repo: "a/b" } };
    expect(followUpMessage(setup, [])).toBe(
      'I\'ve configured the required credentials. Run with these inputs: {\n  "repo": "a/b"\n}',
    );
  });
});

describe("cardIntro", () => {
  const accounts = [{ id: "a", title: "octocat" }];

  function introFor(status: RowStatus) {
    return cardIntro([{ name: "GitHub", status }]);
  }

  it("says nothing once every row is connected", () => {
    expect(cardIntro([])).toBeNull();
    expect(introFor({ kind: "connected", credentialID: "a", label: null })).toBeNull();
  });

  it("asks for all of them when more than one is open", () => {
    expect(
      cardIntro([
        { name: "GitHub", status: { kind: "connect", method: "oauth" } },
        { name: "Linear", status: { kind: "reconnect", credentialID: "b" } },
        { name: "Notion", status: { kind: "connected", credentialID: "c", label: null } },
      ]),
    ).toBe("Connect these services to continue.");
  });

  it("names the one open row and what it needs", () => {
    expect(introFor({ kind: "allow", accounts })).toBe(
      "Allow access to your GitHub account to continue.",
    );
    expect(introFor({ kind: "choose", accounts })).toBe("Choose which GitHub account to use.");
    expect(introFor({ kind: "reconnect", credentialID: "a" })).toBe("Reconnect GitHub to continue.");
  });

  it("matches the way a service gets connected", () => {
    expect(introFor({ kind: "connect", method: "oauth" })).toBe("Sign in to GitHub to continue.");
    expect(introFor({ kind: "connect", method: "mcp" })).toBe("Sign in to GitHub to continue.");
    expect(introFor({ kind: "connect", method: "api_key" })).toBe(
      "Add your GitHub key to continue.",
    );
    expect(introFor({ kind: "connect", method: "user_password" })).toBe(
      "Add your GitHub login to continue.",
    );
    expect(introFor({ kind: "connect", method: "web" })).toBe(
      "Finish connecting GitHub on the web.",
    );
    expect(introFor({ kind: "connect", method: "host_scoped" })).toBe("Connect GitHub to continue.");
  });

  it("asks for a token once a failed sign-in turned the row into one", () => {
    const rows = [
      { name: "Linear", status: { kind: "connect", method: "mcp" } as const, usesToken: true },
    ];
    expect(cardIntro(rows)).toBe("Add a token for Linear to continue.");
  });
});
