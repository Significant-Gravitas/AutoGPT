import { useState } from "react";
import { createRoot } from "react-dom/client";
import * as Atom from "@/components/atoms/Tooltip/BaseTooltip";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import "@/app/globals.css";

function TooltipFixture({ variant }: { variant: string }) {
  if (variant === "information") {
    return (
      <InformationTooltip description="Read the [setup guide](/tooltip-link-destination) before continuing." />
    );
  }
  const { TooltipProvider, Tooltip, TooltipTrigger, TooltipContent } = Atom;
  const content = (
    <TooltipContent side="bottom">An overlapping tooltip</TooltipContent>
  );
  return (
    <TooltipProvider>
      <Tooltip open>
        <TooltipTrigger>Tooltip trigger</TooltipTrigger>
        {/* "ui" is what the deleted components/ui/tooltip rendered: the
            atom's content in a portal. */}
        {variant === "ui" || variant === "atom-portal" ? (
          <Atom.TooltipPortal>{content}</Atom.TooltipPortal>
        ) : (
          content
        )}
      </Tooltip>
    </TooltipProvider>
  );
}

function Fixture() {
  const params = new URLSearchParams(location.search);
  const [clicks, setClicks] = useState(0);
  const content = (
    <div className="relative mx-auto mt-40 w-96">
      <button
        type="button"
        data-testid="underlying"
        className="fixed left-0 top-0"
        onClick={() => setClicks(clicks + 1)}
      >
        Underlying control
      </button>
      <output aria-label="Underlying clicks">{clicks}</output>
      <TooltipFixture variant={params.get("variant") ?? "ui"} />
    </div>
  );
  return params.has("modal") ? (
    <Dialog title="Tooltip regression" forceOpen>
      <Dialog.Content>{content}</Dialog.Content>
    </Dialog>
  ) : (
    content
  );
}

createRoot(document.getElementById("root")!).render(<Fixture />);
