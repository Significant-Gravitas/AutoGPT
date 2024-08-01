import {Card, CardContent} from "@/components/ui/card";
import {Tooltip, TooltipContent, TooltipTrigger} from "@/components/ui/tooltip";
import {Button} from "@/components/ui/button";
import {cn} from "@/lib/utils";
import {Play, Workflow} from "lucide-react";
import SaveDialog from "@/components/modals/SaveDialog";
import {Separator} from "@/components/ui/separator";
import React from "react";

type Action = {
    icon: React.ReactNode;
    label: string;
    onClick: () => void;
}


interface ActionPanelProps {
    actions: Action[];
    onAction: (action: Action) => void;
    onActionPanelClose: () => void;
}

export const ActionPanel= (props: ActionPanelProps) => {

    const { actions, onAction, onActionPanelClose } = props;

    return (
        <aside className="hidden w-14 flex-col sm:flex">
            <Card>
                <CardContent className="p-0">
                    <div className="flex flex-col items-center gap-4 px-2 sm:py-5 rounded-radius">
                        {actions.map((action, index) => (
                            <Tooltip key={index}>
                                <TooltipTrigger asChild>
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        onClick={() => onAction(action)}
                                    >
                                        {action.icon}
                                        <span className="sr-only">{action.label}</span>
                                    </Button>
                                </TooltipTrigger>
                                <TooltipContent side="right">{action.label}</TooltipContent>
                            </Tooltip>
                        ))}
                    </div>
                    <div className="mt-auto flex flex-col items-center gap-4 px-2 sm:py-5">
                        <Separator />
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Button
                                    size="icon"
                                    variant="ghost"
                                    onClick={onActionPanelClose}
                                >
                                    <Play />
                                    <span className="sr-only">Close</span>
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent side="right">Close</TooltipContent>
                        </Tooltip>
                    </div>
                </CardContent>
            </Card>
        </aside>
    );

    // return (
    //     <aside className="hidden w-14 flex-col sm:flex">
    //         <Card>
    //             <CardContent className="p-0">
    //                 <div className="flex flex-col items-center gap-4 px-2 sm:py-5 rounded-radius">
    //                     <Tooltip>
    //                         <TooltipTrigger asChild>
    //                             <Button
    //                                 variant="ghost"
    //                                 size="icon"
    //                                 onClick={() => setIsSidebarOpen(true)}
    //                                 className={cn(isSideBarOpen ? "bg-accent" : "")}
    //                             >
    //                                 <Workflow />
    //                                 <span className="sr-only">Nodes</span>
    //                             </Button>
    //                         </TooltipTrigger>
    //                         <TooltipContent side="right">Add Nodes</TooltipContent>
    //                     </Tooltip>
    //                     <Tooltip>
    //                         <TooltipContent side="right">Save</TooltipContent>
    //                         <TooltipTrigger asChild>
    //                             <SaveDialog />
    //                         </TooltipTrigger>
    //                     </Tooltip>
    //                 </div>
    //                 <div className="mt-auto flex flex-col items-center gap-4 px-2 sm:py-5">
    //                     <Separator />
    //                     <Tooltip>
    //                         <TooltipTrigger asChild>
    //                             <Button
    //                                 size="icon"
    //                                 variant="ghost"
    //                             >
    //                                 <Play />
    //                                 <span className="sr-only">Save and Run</span>
    //                             </Button>
    //                         </TooltipTrigger>
    //                         <TooltipContent side="right">Save and Run</TooltipContent>
    //                     </Tooltip>
    //                 </div>
    //             </CardContent>
    //         </Card>
    //     </aside>
    // );
}
export default ActionPanel;