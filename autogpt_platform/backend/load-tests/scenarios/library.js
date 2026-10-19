// The library page: the account's agents, a search over them, and one
// agent's detail. Under 1% of production requests, but the list and the
// detail are the slowest common routes there, so the journey gets more
// than its share of the run.
import exec from "k6/execution";

import { iterationPhase } from "../configs/profiles.js";
import { api } from "../lib/http.js";
import { iterationRandom, pick } from "../lib/random.js";

export function libraryBrowse(data, profile) {
  const phase = iterationPhase(profile);
  const random = iterationRandom(
    exec.scenario.name,
    exec.scenario.iterationInTest,
  );
  const account = data.accounts[(__VU - 1) % data.accounts.length];

  const list = api(
    "GET",
    "/library/agents?page=1&page_size=15&sort_by=updatedAt",
    {
      data,
      endpoint: "library.list",
      phase,
    },
  );
  api("GET", "/library/agents?search_term=load%20test&page=1&page_size=15", {
    data,
    route: "/library/agents",
    endpoint: "library.search",
    phase,
  });

  const listed =
    list.json && Array.isArray(list.json.agents) ? list.json.agents : [];
  const agentId =
    pick(random, account.libraryAgentIds) ||
    (listed[0] && listed[0].id) ||
    null;
  if (agentId) {
    api("GET", `/library/agents/${agentId}`, {
      data,
      route: "/library/agents/{library_agent_id}",
      endpoint: "library.agent_detail",
      phase,
    });
  }
}
