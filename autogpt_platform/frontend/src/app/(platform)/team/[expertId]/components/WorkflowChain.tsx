"use client";
import { ExpertWorkflowChainItem } from "@/app/api/__generated__/models/expertWorkflowChainItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ArrowRight01Icon, FlashIcon } from "@hugeicons/core-free-icons";
import { Fragment } from "react";
import { ChainTile } from "./ChainTile";

interface Props {
  chain: ExpertWorkflowChainItem[];
  size?: "sm" | "md";
}

export function WorkflowChain({ chain, size = "md" }: Props) {
  if (chain.length === 0) {
    return (
      <span className="inline-flex text-foreground-disabled">
        <Icon icon={FlashIcon} size={size === "sm" ? 20 : 28} />
      </span>
    );
  }

  return (
    <>
      <div className="flex items-center gap-3" data-testid="workflow-chain">
        {chain.map((item, index) => (
          <Fragment key={`${item.kind}-${item.provider ?? index}`}>
            {index > 0 ? (
              <span className="inline-flex text-foreground-emphasis">
                <Icon icon={ArrowRight01Icon} size={14} />
              </span>
            ) : null}
            <ChainTile item={item} size={size} />
          </Fragment>
        ))}
      </div>
    </>
  );
}
