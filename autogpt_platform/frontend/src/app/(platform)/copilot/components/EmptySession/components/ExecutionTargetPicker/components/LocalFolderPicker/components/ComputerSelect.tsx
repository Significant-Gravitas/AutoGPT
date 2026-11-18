"use client";

import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";
import { Button } from "@/components/atoms/Button/Button";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { LocalExecutorSetup } from "../../LocalExecutorSetup/LocalExecutorSetup";
import { computerOptions, hasComputersWithOneName } from "../helpers";

const SETUP_PANEL_ID = "local-executor-another-computer";

interface Props {
  machines: ExecutorMachine[];
  selectedMachineID: string | null;
  isRefreshingMachines: boolean;
  isSetupOpen: boolean;
  connectedNotice: string;
  onSelectMachine: (machineID: string) => void;
  onRefreshMachines: () => void;
  onToggleSetup: () => void;
}

export function ComputerSelect({
  machines,
  selectedMachineID,
  isRefreshingMachines,
  isSetupOpen,
  connectedNotice,
  onSelectMachine,
  onRefreshMachines,
  onToggleSetup,
}: Props) {
  return (
    <div className="flex flex-col gap-2">
      <Select
        id="local-executor-machine"
        label="Computer"
        value={selectedMachineID ?? ""}
        onValueChange={onSelectMachine}
        options={computerOptions(machines)}
        wrapperClassName="!mb-0"
      />
      {hasComputersWithOneName(machines) ? (
        <Text variant="small" tone="secondary">
          Some of your computers have the same name. To tell them apart, set{" "}
          <span className="font-mono text-caption">display_name</span> in each
          one&apos;s config.toml and start it again.
        </Text>
      ) : null}
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <Button
          type="button"
          variant="link"
          size="inline"
          aria-expanded={isSetupOpen}
          aria-controls={SETUP_PANEL_ID}
          onClick={onToggleSetup}
        >
          Connect Another Computer
        </Button>
        <Text role="status" variant="small" tone="success-strong">
          {connectedNotice}
        </Text>
      </div>
      <div id={SETUP_PANEL_ID} hidden={!isSetupOpen}>
        {isSetupOpen ? (
          <LocalExecutorSetup
            isRefreshing={isRefreshingMachines}
            onRefresh={onRefreshMachines}
          />
        ) : null}
      </div>
    </div>
  );
}
