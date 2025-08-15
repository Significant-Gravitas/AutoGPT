import { Page } from "@playwright/test";
import { BasePage } from "./base.page";
import path from "path";

interface Agent {
  id: string;
  name: string;
  runCount: number;
  lastRun: string;
}

interface Run {
  id: string;
  agentId: string;
  agentName: string;
  started: string;
  duration: number;
  status: string;
}

interface Schedule {
  id: string;
  graphName: string;
  nextExecution: string;
  schedule: string;
  actions: string[];
}

enum ImportType {
  AGENT = "agent",
  TEMPLATE = "template",
}

export class MonitorPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  async isLoaded(): Promise<boolean> {
    // Wait for the monitor page
    await this.page.getByTestId("monitor-page").waitFor({
      state: "visible",
      timeout: 10_000,
    });

    // Wait for table headers to be visible (indicates table structure is ready)
    await this.page.locator("thead th").first().waitFor({
      state: "visible",
      timeout: 15_000,
    });

    // Wait for either a table row or an empty tbody to be present
    await Promise.race([
      // Wait for at least one row
      this.page.locator("tbody tr[data-testid]").first().waitFor({
        state: "visible",
        timeout: 15_000,
      }),
      // OR wait for an empty tbody (indicating no agents but table is loaded)
      this.page
        .locator("tbody[data-testid='agent-flow-list-body']:empty")
        .waitFor({
          state: "visible",
          timeout: 15_000,
        }),
    ]);

    return true;
  }

  async listAgents(): Promise<Agent[]> {
    // Wait for table rows to be available
    const rows = await this.page.locator("tbody tr[data-testid]").all();

    const agents: Agent[] = [];

    for (const row of rows) {
      // Get the id from data-testid attribute
      const id = (await row.getAttribute("data-testid")) || "";

      // Get columns - there are 3 cells per row (name, run count, last run)
      const cells = await row.locator("td").all();

      // Extract name from first cell
      const name = (await row.getAttribute("data-name")) || "";

      // Extract run count from second cell
      const runCountText = (await cells[1].textContent()) || "0";
      const runCount = parseInt(runCountText, 10);

      // Extract last run from third cell's title attribute (contains full timestamp)
      // If no title, the cell will be empty indicating no last run
      const lastRunCell = cells[2];
      const lastRun = (await lastRunCell.getAttribute("title")) || "";

      agents.push({
        id,
        name,
        runCount,
        lastRun,
      });
    }

    agents.reduce((acc, agent) => {
      if (!agent.id.includes("flow-run")) {
        acc.push(agent);
      }
      return acc;
    }, [] as Agent[]);

    return agents;
  }

  async listRuns(filter?: Agent): Promise<Run[]> {
    // Wait for the runs table to be loaded - look for table header "Agent"
    await this.page.locator("[data-testid='flow-runs-list-body']").waitFor({
      timeout: 10000,
    });

    // Get all run rows
    const rows = await this.page
      .locator('tbody tr[data-testid^="flow-run-"]')
      .all();

    const runs: Run[] = [];

    for (const row of rows) {
      const runId = (await row.getAttribute("data-runid")) || "";
      const agentId = (await row.getAttribute("data-graphid")) || "";

      // Get columns
      const cells = await row.locator("td").all();

      // Parse data from cells
      const agentName = (await cells[0].textContent()) || "";
      const started = (await cells[1].textContent()) || "";
      const status = (await cells[2].locator("div").textContent()) || "";
      const duration = (await cells[3].textContent()) || "";

      // Only add if no filter or if matches filter
      if (!filter || filter.id === agentId) {
        runs.push({
          id: runId,
          agentId: agentId,
          agentName: agentName.trim(),
          started: started.trim(),
          duration: parseFloat(duration.replace("s", "")),
          status: status.toLowerCase().trim(),
        });
      }
    }

    return runs;
  }
  async listSchedules(): Promise<Schedule[]> {
    return [];
  }

  async clickAgent(id: string) {
    await this.page.getByTestId(id).click();
  }

  async clickCreateAgent(): Promise<void> {
    await this.page.getByRole("link", { name: "Create" }).click();
  }

  async importFromFile(
    directory: string,
    file: string,
    name?: string,
    description?: string,
    importType: ImportType = ImportType.AGENT,
  ) {
    await this.page.getByTestId("create-agent-dropdown").click();
    await this.page.getByTestId("import-agent-from-file").click();

    await this.page
      .getByTestId("import-agent-file-input")
      .setInputFiles(path.join(directory, file));
    if (name) {
      await this.page.getByTestId("agent-name-input").fill(name);
    }
    if (description) {
      await this.page.getByTestId("agent-description-input").fill(description);
    }
    if (importType === ImportType.TEMPLATE) {
      await this.page.getByTestId("import-as-template-switch").click();
    }
    await this.page.getByTestId("import-agent-submit").click();
  }

  async deleteAgent() {
    await this.page.getByTestId("delete-agent-button").click();
  }

  async exportToFile(agent: Agent) {
    await this.clickAgent(agent.id);
    await this.page.getByTestId("export-button").click();
  }
}
