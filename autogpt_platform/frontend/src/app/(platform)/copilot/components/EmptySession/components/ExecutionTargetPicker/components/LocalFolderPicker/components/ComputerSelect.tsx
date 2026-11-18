"use client";

import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";
import { Button } from "@/components/atoms/Button/Button";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { useEffect, useRef } from "react";
import { LocalExecutorSetup } from "../../LocalExecutorSetup/LocalExecutorSetup";
import { computerMenu } from "../helpers";

const MENU_ID = "local-executor-machine";
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
  const { options, hasSharedNames } = computerMenu(machines);
  const hasFocusInSetup = useRef(false);

  // The steps close by themselves when another computer connects. Focus that
  // was in them (on "Check Again", say) would be dropped with them.
  useEffect(
    function focusTheMenuWhenTheStepsCloseUnderFocus() {
      if (isSetupOpen || !hasFocusInSetup.current) return;
      hasFocusInSetup.current = false;
      document.getElementById(MENU_ID)?.focus();
    },
    [isSetupOpen],
  );

  return (
    <div className="flex flex-col gap-2">
      <Select
        id={MENU_ID}
        label="Computer"
        value={selectedMachineID ?? ""}
        onValueChange={onSelectMachine}
        options={options}
        wrapperClassName="!mb-0"
      />
      {hasSharedNames ? (
        <Text variant="small" tone="secondary">
          Some of your computers have the same name. To tell them apart, set{" "}
          <span className="font-mono text-caption">display_name</span> in each
          one&apos;s config.toml and start it again. The steps under Connect
          Another Computer list where that file is.
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
        <Text
          variant="small"
          tone="success-strong"
          className="min-w-0 break-words"
        >
          {connectedNotice}
        </Text>
      </div>
      <div
        id={SETUP_PANEL_ID}
        hidden={!isSetupOpen}
        onFocus={() => {
          hasFocusInSetup.current = true;
        }}
        onBlur={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget)) {
            hasFocusInSetup.current = false;
          }
        }}
      >
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
