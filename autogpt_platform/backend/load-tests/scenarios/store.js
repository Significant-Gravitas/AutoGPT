// Browsing the marketplace, as a visitor does without signing in: the skills
// list and a skill's page, the store agents list and an agent's page, a
// search, the creators list; then, signed in, the account's store profile.
//
// The items come from the target's own listings, found by the seed step.
// Searches cycle through a fixed set of queries: the backend caches each
// query's result for five minutes, so a fixed set keeps its embedding calls
// to a handful per run.
import exec from "k6/execution";

import { iterationPhase } from "../configs/profiles.js";
import { api } from "../lib/http.js";
import { iterationRandom, pick } from "../lib/random.js";

const SEARCH_QUERIES = [
  "email",
  "research",
  "summarize news",
  "calendar",
  "social media posts",
  "lead generation",
];

export function storeBrowse(data, profile) {
  const phase = iterationPhase(profile);
  const random = iterationRandom(
    exec.scenario.name,
    exec.scenario.iterationInTest,
  );
  const store = data.store;

  if (store.skillsAvailable) {
    api("GET", "/store/skills?page=1&page_size=20", {
      auth: false,
      endpoint: "store.skills_list",
      phase,
    });
    const skill = pick(random, store.skillSlugs);
    if (skill) {
      api("GET", `/store/skills/${encodeURIComponent(skill)}`, {
        auth: false,
        route: "/store/skills/{slug}",
        endpoint: "store.skill_detail",
        phase,
      });
    }
  }

  api("GET", "/store/agents?page=1&page_size=20&sorted_by=runs", {
    auth: false,
    endpoint: "store.agents_list",
    phase,
  });
  const agent = pick(random, store.agents);
  if (agent) {
    api(
      "GET",
      `/store/agents/${encodeURIComponent(agent.creator)}/${encodeURIComponent(agent.slug)}`,
      {
        auth: false,
        route: "/store/agents/{username}/{agent_name}",
        endpoint: "store.agent_detail",
        phase,
      },
    );
  }
  api(
    "GET",
    `/store/agents?search_query=${encodeURIComponent(pick(random, SEARCH_QUERIES))}&page=1&page_size=20`,
    {
      auth: false,
      route: "/store/agents",
      endpoint: "store.agents_search",
      phase,
    },
  );
  api("GET", "/store/creators?page=1&page_size=20&sorted_by=agent_rating", {
    auth: false,
    endpoint: "store.creators",
    phase,
  });
  api("GET", "/store/profile", { data, endpoint: "store.profile", phase });
}
