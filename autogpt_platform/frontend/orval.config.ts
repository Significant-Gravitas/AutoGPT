import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { defineConfig } from "orval";

const ENDPOINTS_DIR = "./src/app/api/__generated__/endpoints";

// Orval 8 starts every infinite query key with "infinite"; orval 7 did not.
// The app invalidates and updates infinite lists through the plain query key
// (getGetV1ListGraphExecutionsQueryKey and friends), so restore the old keys.
function dropInfiniteQueryKeyPrefix() {
  const pattern =
    /(InfiniteQueryKey = \([\s\S]*?\) => \{\s*return \[\s*)["']infinite["'],\s*/g;
  let fixed = 0;
  for (const file of readdirSync(ENDPOINTS_DIR, { recursive: true })) {
    if (typeof file !== "string" || !file.endsWith(".ts")) continue;
    const path = join(ENDPOINTS_DIR, file);
    const source = readFileSync(path, "utf8");
    const result = source.replace(pattern, (_match, keyStart: string) => {
      fixed += 1;
      return keyStart;
    });
    if (result !== source) writeFileSync(path, result);
  }
  if (fixed === 0) {
    console.error("orval: found no infinite query keys to fix");
    process.exitCode = 1;
  }
}

export default defineConfig({
  autogpt_api_client: {
    input: {
      target: `./src/app/api/openapi.json`,
      override: {
        transformer: "./src/app/api/transformers/fix-spec.mjs",
      },
    },
    output: {
      workspace: "./src/app/api",
      target: `./__generated__/endpoints`,
      schemas: "./__generated__/models",
      mode: "tags-split",
      client: "react-query",
      httpClient: "fetch",
      indexFiles: false,
      mock: {
        generators: [
          {
            type: "msw",
            baseUrl: "/api/proxy",
            generateEachHttpStatus: true,
            delay: 0,
          },
        ],
      },
      override: {
        mutator: {
          path: "./mutators/custom-mutator.ts",
          name: "customMutator",
        },
        query: {
          usePrefetch: true,
          // Will add more as their use cases arise
        },
        useDates: true,
        operations: {
          "getV2List library agents": {
            query: {
              useInfinite: true,
              useInfiniteQueryParam: "page",
            },
          },
          "getV2List marketplace skills": {
            query: {
              useInfinite: true,
              useInfiniteQueryParam: "page",
            },
          },
          "getV2List favorite library agents": {
            query: {
              useInfinite: true,
              useInfiniteQueryParam: "page",
            },
          },
          "getV2List presets": {
            query: {
              useInfinite: true,
              useInfiniteQueryParam: "page",
            },
          },
          "getV1List graph executions": {
            query: {
              useInfinite: true,
              useInfiniteQueryParam: "page",
            },
          },
          "getV2Get builder blocks": {
            query: {
              useInfinite: true,
              useInfiniteQueryParam: "page",
              useQuery: true,
            },
          },
          "getV2Get builder integration providers": {
            query: {
              useInfinite: true,
              useInfiniteQueryParam: "page",
            },
          },
          "getV2List store agents": {
            query: {
              useInfinite: true,
              useInfiniteQueryParam: "page",
              useQuery: true,
            },
          },
          "getV2Builder search": {
            query: {
              useInfinite: true,
              useInfiniteQueryParam: "page",
            },
          },
        },
      },
    },
    hooks: {
      afterAllFilesWrite: [
        dropInfiniteQueryKeyPrefix,
        "prettier --ignore-path= --write ./src/app/api/__generated__",
      ],
    },
  },
  // autogpt_zod_schema: {
  //   input: {
  //     target: `./src/app/api/openapi.json`,
  //     override: {
  //       transformer: "./src/app/api/transformers/fix-spec.mjs",
  //     },
  //   },
  //   output: {
  //     workspace: "./src/app/api",
  //     target: `./__generated__/zod-schema`,
  //     schemas: "./__generated__/models",
  //     mode: "tags-split",
  //     client: "zod",
  //     indexFiles: false,
  //   },
  //   hooks: {
  //     afterAllFilesWrite: "prettier --write",
  //   },
  // },
});
