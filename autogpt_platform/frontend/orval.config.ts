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
      target: `./__gen__/endpoints`,
      schemas: "./__gen__/models",
      mode: "tags-split",
      client: "react-query",
      httpClient: "fetch",
      indexFiles: false,
      mock: {
        type: "msw",
        delay: 1000, // artifical latency
        generateEachHttpStatus: true, // helps us test error-handling scenarios and generate mocks for all HTTP statuses
      },
      override: {
        mutator: {
          path: "./mutators/custom-mutator.ts",
          name: "customMutator",
        },
        query: {
          useQuery: true,
          useMutation: true,
          // Will add more as their use cases arise
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
      target: `./__gen__/zod-schema`,
      schemas: "./__gen__/models",
      mode: "tags-split",
      client: "zod",
      indexFiles: false,
    },
    hooks: {
      afterAllFilesWrite: "prettier --write",
    },
  },
});
