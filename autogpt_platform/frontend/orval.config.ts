import { defineConfig } from "orval";

export default defineConfig({
  autogpt_api_client: {
    input: {
      target: `http://localhost:8006/openapi.json`,
      override: {
        transformer: "./src/api/transformers/fix-tags.js",
      },
    },
    output: {
      workspace: "./src/api",
      target: `./endpoints`,
      schemas: "./model",
      mode: "tags-split",
      client: "react-query",
      httpClient: "fetch",
      mock: {
        type: "msw",
        delay: 1000,
      },
      override: {
        mutator: {
          path: "./mutators/custom-mutator.ts",
          name: "customMutator",
        },
        query: {
          useQuery: true,
          useInfinite: true,
          useInfiniteQueryParam: "nextId",
        },
      },
    },
    hooks: {
      afterAllFilesWrite: "prettier --write",
    },
  },
  autogpt_zod_schema: {
    input: {
      target: `http://localhost:8006/openapi.json`,
      override: {
        transformer: "./src/api/transformers/fix-tags.js",
      },
    },
    output: {
      workspace: "./src/api",
      target: `./zod-schema`,
      schemas: "./model",
      mode: "tags-split",
      client: "zod",
    },
    hooks: {
      afterAllFilesWrite: "prettier --write",
    },
  },
});
