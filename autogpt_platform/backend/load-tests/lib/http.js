// One way to call the API from every scenario, so metrics line up:
//
// - every request carries an `endpoint` tag (a short stable name such as
//   store.skill_detail) and a `phase` tag (warmup or steady); thresholds and
//   the summary group by them;
// - the `name` tag is the route template, without the query, so k6 groups a
//   thousand /api/store/skills/<slug> requests under one name;
// - each request makes one check, named after the endpoint, that passes when
//   the status is one the caller expects.
import { check } from "k6";
import http from "k6/http";

import { getEnvironmentConfig } from "../configs/environment.js";
import { authHeaders } from "./auth.js";

const config = getEnvironmentConfig();

// GET/POST/DELETE a backend route. `path` is relative to /api; `route` is the
// template the metrics are grouped under (defaults to path without its
// query).
export function api(method, path, options) {
  const opts = options || {};
  const headers = Object.assign(
    { "Content-Type": "application/json" },
    opts.auth === false ? {} : authHeaders(opts.data),
    opts.headers || {},
  );
  const expected = opts.expect || [200];
  const params = {
    headers,
    tags: {
      name: `/api${opts.route || path.split("?")[0]}`,
      endpoint: opts.endpoint,
      phase: opts.phase || "steady",
    },
    responseCallback: http.expectedStatuses(...expected),
    timeout: opts.timeout || "60s",
  };
  const body = opts.body === undefined ? null : JSON.stringify(opts.body);
  const res = http.request(
    method,
    `${config.API_BASE_URL}${path}`,
    body,
    params,
  );

  const ok = expected.includes(res.status);
  check(
    res,
    {
      [`${opts.endpoint}: status ${expected.join("/")}`]: () => ok,
    },
    { endpoint: opts.endpoint, phase: opts.phase || "steady" },
  );
  if (!ok && opts.quiet !== true) {
    console.warn(
      `${opts.endpoint}: ${method} ${path} answered HTTP ${res.status}${
        res.status === 0 ? ` (${res.error})` : ""
      }`,
    );
  }
  return { res, ok, json: ok ? parseJson(res) : null };
}

export function parseJson(res) {
  try {
    return res.json();
  } catch (e) {
    return null;
  }
}

// The frontend origin for routes that live there (Better Auth).
export function frontendUrl() {
  return config.FRONTEND_URL;
}
