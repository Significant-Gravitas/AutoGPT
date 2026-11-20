import type { ConnectableService } from "../../../listHelpers";
import { serverTarget, sheetLists } from "../helpers";

const services: ConnectableService[] = [
  {
    key: "github",
    name: "GitHub",
    description: "Code hosting",
    providers: ["github"],
    action: { kind: "oauth", provider: "github" },
  },
  {
    key: "google",
    name: "Google",
    description: null,
    providers: ["google"],
    action: { kind: "oauth", provider: "google" },
  },
];
const accounts = [{ id: "a", name: "Linear", detail: "AutoGPT", logoID: "linear" }];

describe("sheetLists", () => {
  it("lists everything with no query", () => {
    const lists = sheetLists({ services, accounts, query: "" });
    expect(lists.accounts).toHaveLength(1);
    expect(lists.services).toHaveLength(2);
    expect(lists.showServer).toBe(true);
    expect(lists.isEmpty).toBe(false);
  });

  it("matches names, descriptions and account details", () => {
    expect(sheetLists({ services, accounts, query: "code" }).services.map((s) => s.key)).toEqual(["github"]);
    expect(sheetLists({ services, accounts, query: "autogpt" }).accounts).toHaveLength(1);
    expect(sheetLists({ services, accounts, query: "mcp" }).showServer).toBe(true);
  });

  it("says when nothing matches", () => {
    const lists = sheetLists({ services, accounts, query: "zzz" });
    expect(lists).toMatchObject({ accounts: [], services: [], showServer: false, isEmpty: true });
  });
});

describe("serverTarget", () => {
  it("reads an http(s) address and names it by host", () => {
    expect(serverTarget(" https://mcp.example.com/mcp/ ")).toEqual({
      serverURL: "https://mcp.example.com/mcp",
      providerName: "mcp.example.com",
    });
  });

  it("refuses anything else", () => {
    expect(serverTarget("")).toBeNull();
    expect(serverTarget("mcp.example.com")).toBeNull();
    expect(serverTarget("ftp://mcp.example.com")).toBeNull();
    expect(serverTarget("https://mcp example.com")).toBeNull();
    expect(serverTarget("https://:8080")).toBeNull();
  });
});
