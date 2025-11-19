import { useReactFlow } from "@xyflow/react";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import {
  FrameCornersIcon,
  MinusIcon,
  PlusIcon,
} from "@phosphor-icons/react/dist/ssr";
import { LockIcon, LockOpenIcon } from "lucide-react";
import { memo } from "react";

export const CustomControls = memo(
  ({
    setIsLocked,
    isLocked,
  }: {
    isLocked: boolean;
    setIsLocked: (isLocked: boolean) => void;
  }) => {
    const { zoomIn, zoomOut, fitView } = useReactFlow();

    const controls = [
      {
        icon: <PlusIcon className="size-4" />,
        label: "Zoom In",
        onClick: () => zoomIn(),
        className: "h-10 w-10 border-none",
      },
      {
        icon: <MinusIcon className="size-4" />,
        label: "Zoom Out",
        onClick: () => zoomOut(),
        className: "h-10 w-10 border-none",
      },
      {
        icon: <FrameCornersIcon className="size-4" />,
        label: "Fit View",
        onClick: () => fitView({ padding: 0.2, duration: 800, maxZoom: 1 }),
        className: "h-10 w-10 border-none",
      },
      {
        icon: !isLocked ? (
          <LockOpenIcon className="size-4" />
        ) : (
          <LockIcon className="size-4" />
        ),
        label: "Toggle Lock",
        onClick: () => setIsLocked(!isLocked),
        className: `h-10 w-10 border-none ${isLocked ? "bg-zinc-100" : "bg-white"}`,
      },
    ];

    return (
      <div className="absolute bottom-4 left-4 z-10 flex flex-col items-center gap-2 rounded-full bg-white px-1 py-2 shadow-lg">
        {controls.map((control, index) => (
          <Tooltip key={index} delayDuration={300}>
            <TooltipTrigger asChild>
              <Button
                variant="icon"
                size={"small"}
                onClick={control.onClick}
                className={control.className}
              >
                {control.icon}
                <span className="sr-only">{control.label}</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">{control.label}</TooltipContent>
          </Tooltip>
        ))}
      </div>
    );
  },
);

CustomControls.displayName = "CustomControls";
