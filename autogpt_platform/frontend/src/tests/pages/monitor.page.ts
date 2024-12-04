import { ElementHandle, Locator, Page } from "@playwright/test";
import { BasePage } from "./base.page";

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

interface AgentRun extends Agent {
  runs: Run[];
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
    console.log(`checking if monitor page is loaded`);
    try {
      // Wait for network to settle first
      await this.page.waitForLoadState("networkidle", { timeout: 10_000 });

      // Wait for the monitor page container
      await this.page.getByTestId("monitor-page").waitFor({
        state: "visible",
        timeout: 10_000,
      });

      // Wait for table headers to be visible (indicates table structure is ready)
      await this.page.locator("thead th").first().waitFor({
        state: "visible",
        timeout: 5_000,
      });

      // Wait for either a table row or an empty tbody to be present
      await Promise.race([
        // Wait for at least one row
        this.page.locator("tbody tr[data-testid]").first().waitFor({
          state: "visible",
          timeout: 5_000,
        }),
        // OR wait for an empty tbody (indicating no agents but table is loaded)
        this.page
          .locator("tbody[data-testid='agent-flow-list-body']:empty")
          .waitFor({
            state: "visible",
            timeout: 5_000,
          }),
      ]);

      return true;
    } catch (error) {
      return false;
    }
  }

  async listAgents(): Promise<Agent[]> {
    console.log(`listing agents`);
    // Wait for table rows to be available
    const rows = await this.page.locator("tbody tr[data-testid]").all();

    const agents: Agent[] = [];

    for (const row of rows) {
      // Get the id from data-testid attribute
      const id = (await row.getAttribute("data-testid")) || "";

      // Get columns - there are 3 cells per row (name, run count, last run)
      const cells = await row.locator("td").all();

      // Extract name from first cell
      const name = (await cells[0].textContent()) || "";

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

    return agents;
  }

  async listRuns(filter?: Agent): Promise<Run[]> {
    console.log(`listing runs`);
    // Wait for the runs table to be loaded - look for table header "Agent"
    await this.page.locator("[data-testid='flow-runs-list-body']").waitFor();

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
    console.log(`listing schedules`);
    return [];
  }

  async selectAgent(id?: string, name?: string, regex?: string) {
    console.log(`selecting agent ${id} ${name} ${regex}`);
  }

  async clickCreateAgent(): Promise<void> {
    console.log(`clicking create agent`);
    await this.page.getByRole("link", { name: "Create" }).click();
  }

  async importFromFile(
    file: string,
    name?: string,
    description?: string,
    importType: ImportType = ImportType.AGENT,
  ) {
    console.log(
      `importing from file ${file} ${name} ${description} ${importType}`,
    );
  }

  async deleteAgent(agent: Agent) {
    console.log(`deleting agent ${agent.id} ${agent.name}`);
  }

  async clickAllVersions(agent: Agent) {
    console.log(`clicking all versions for agent ${agent.id} ${agent.name}`);
  }

  async openInBuilder(agent: Agent) {
    console.log(`opening agent ${agent.id} ${agent.name} in builder`);
  }

  async exportToFile(agent: Agent) {
    console.log(`exporting agent ${agent.id} ${agent.name} to file`);
  }

  async selectRun(agent: Agent, run: Run) {
    console.log(`selecting run ${run.id} for agent ${agent.id} ${agent.name}`);
  }

  async openOutputs(agent: Agent, run: Run) {
    console.log(
      `opening outputs for run ${run.id} of agent ${agent.id} ${agent.name}`,
    );
  }
}
