import {Card, CardContent} from "@/components/ui/card";
import {Tooltip, TooltipContent, TooltipTrigger} from "@/components/ui/tooltip";
import {Button} from "@/components/ui/button";
import {Separator} from "@/components/ui/separator";
import React from "react";

export type Control = {
    icon: React.ReactNode;
    label: string;
    onClick: () => void;
}

interface ControlPanelProps {
    controls: Control[];
    children?: React.ReactNode;
}

export const ControlPanel= ( {controls, children}: ControlPanelProps) => {
    return (
        <aside className="hidden w-14 flex-col sm:flex">
            <Card>
                <CardContent className="p-0">
                    <div className="flex flex-col items-center gap-4 px-2 sm:py-5 rounded-radius">
                        {controls.map((control, index) => (
                            <Tooltip key={index} delayDuration={500}>
                                <TooltipTrigger asChild>
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        onClick={() => control.onClick()}
                                    >
                                        {control.icon}
                                        <span className="sr-only">{control.label}</span>
                                    </Button>
                                </TooltipTrigger>
                                <TooltipContent side="right">{control.label}</TooltipContent>
                            </Tooltip>
                        ))}
                        <Separator />
                        {children}
                    </div>
                </CardContent>
            </Card>
        </aside>
    );
}
export default ControlPanel;