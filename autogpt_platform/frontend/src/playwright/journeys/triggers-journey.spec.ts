import type { Page } from "@playwright/test";
import { createUniqueAgentName } from "../pages/build.page";
import { backendURL } from "./backend";
import { expect, test } from "./fixtures";

const WEBHOOK_TRIGGER_BLOCK_ID = "8fa8c167-2002-47ce-aba8-97572fc5d387";
const AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4";
const RUN_ENDED = /^(COMPLETED|FAILED|TERMINATED)$/;

interface Graph {
  id: string;
  version: number;
}

interface Run {
  id: string;
  status: string;
}

test.describe("triggers journey", () => {
  test("a webhook trigger runs its agent with what it receives, and only with its secret", async ({
    page,
    playwright,
    journeyUser: _user,
  }) => {
    const backend = backendURL();
    const graph = await createWebhookAgent(
      page,
      createUniqueAgentName("Journey webhook trigger"),
    );
    const secret = `journey-secret-${Date.now()}`;
    const preset = await setUpTrigger(page, graph, secret);
    // Deliveries carry no session, as a stranger's or a service's would, so
    // the secret alone has to admit them.
    const deliveries = await playwright.request.newContext({
      storageState: { cookies: [], origins: [] },
    });

    try {
      expect(
        preset.webhook_id,
        "the trigger should get a webhook",
      ).toBeTruthy();
      const ingress = `${backend}/api/integrations/generic_webhook/webhooks/${preset.webhook_id}/ingress`;

      // Anyone who learns the URL can post to it, so the secret is what
      // keeps a stranger from running the agent.
      const refused = await deliveries.post(ingress, {
        data: { message: "without the secret" },
      });
      expect(refused.status(), "a delivery without the secret").toBe(403);

      const delivered = await deliveries.post(ingress, {
        data: { message: "hello from the journey" },
        headers: { "X-Webhook-Secret": secret },
      });
      expect(
        delivered.ok(),
        `delivering with the secret failed: ${delivered.status()}`,
      ).toBe(true);

      // Exactly one run, the delivery with the secret, and it ends.
      await expect
        .poll(
          async () => (await listRuns(page, graph.id)).map((run) => run.status),
          {
            message: "the delivery should start one run that ends",
            timeout: 60_000,
            intervals: [1_000],
          },
        )
        .toEqual([expect.stringMatching(RUN_ENDED)]);
      const run = (await listRuns(page, graph.id)).at(0);
      if (!run) throw new Error("the run is no longer listed");
      expect(run.status).toBe("COMPLETED");

      // The run saw what was delivered.
      const details = await page.request.get(
        `/api/proxy/api/graphs/${graph.id}/executions/${run.id}`,
      );
      expect(details.ok(), `reading the run: ${details.status()}`).toBe(true);
      const { outputs } = (await details.json()) as {
        outputs: Record<string, unknown[]>;
      };
      expect(outputs.payload).toEqual([
        expect.objectContaining({ message: "hello from the journey" }),
      ]);
    } finally {
      await deliveries.dispose();
      // Cleanup records its own failure, never throwing over the one that
      // brought the test here.
      try {
        const removed = await page.request.delete(
          `/api/proxy/api/library/presets/${preset.id}`,
        );
        expect
          .soft(
            removed.ok(),
            `deleting the trigger failed: ${removed.status()}`,
          )
          .toBe(true);
      } catch (error) {
        expect.soft(error, "deleting the trigger threw").toBeUndefined();
      }
    }
  });
});

// An agent whose webhook trigger hands what it receives to an output.
async function createWebhookAgent(page: Page, name: string): Promise<Graph> {
  const response = await page.request.post("/api/proxy/api/graphs", {
    data: {
      source: "upload",
      graph: {
        name,
        description:
          "Records what a webhook delivers, for the triggers journey",
        is_active: true,
        nodes: [
          {
            id: "trigger",
            block_id: WEBHOOK_TRIGGER_BLOCK_ID,
            input_default: {},
            metadata: { position: { x: 0, y: 0 } },
            input_links: [],
            output_links: [],
          },
          {
            id: "output",
            block_id: AGENT_OUTPUT_BLOCK_ID,
            input_default: { name: "payload" },
            metadata: { position: { x: 400, y: 0 } },
            input_links: [],
            output_links: [],
          },
        ],
        links: [
          {
            source_id: "trigger",
            source_name: "payload",
            sink_id: "output",
            sink_name: "value",
          },
        ],
      },
    },
  });
  expect(response.ok(), `creating the agent failed: ${response.status()}`).toBe(
    true,
  );
  const { id, version } = (await response.json()) as Graph;
  return { id, version };
}

async function setUpTrigger(page: Page, graph: Graph, secret: string) {
  const response = await page.request.post(
    "/api/proxy/api/library/presets/setup-trigger",
    {
      data: {
        name: "Journey trigger",
        description: "",
        graph_id: graph.id,
        graph_version: graph.version,
        trigger_config: { secret_token: secret },
        constant_inputs: {},
        agent_credentials: {},
      },
    },
  );
  expect(
    response.ok(),
    `setting up the trigger failed: ${response.status()}`,
  ).toBe(true);
  return (await response.json()) as { id: string; webhook_id: string | null };
}

async function listRuns(page: Page, graphId: string): Promise<Run[]> {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${graphId}/executions`,
  );
  expect(response.ok(), `listing runs: ${response.status()}`).toBe(true);
  return ((await response.json()) as { executions: Run[] }).executions;
}
