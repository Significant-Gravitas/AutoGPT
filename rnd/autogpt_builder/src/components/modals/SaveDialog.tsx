'use client'

import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger
} from "@/components/ui/dialog";
import {Button} from "@/components/ui/button";
import {Save} from "lucide-react";
import {Label} from "@/components/ui/label";
import {Input} from "@/components/ui/input";
import React from "react";
import {Tooltip, TooltipContent, TooltipTrigger} from "@/components/ui/tooltip";
import {GraphMeta} from "@/lib/autogpt-server-api";

interface SaveDialogProps {
    agentMeta: GraphMeta | null;
    onSave: (isTemplate: boolean | undefined) => void;
    onNameChange: (name: string) => void;
    onDescriptionChange: (description: string) => void;
}

export const SaveDialog = ({ agentMeta, onSave, onNameChange, onDescriptionChange }: SaveDialogProps) => {
    const handleSave = () => {
        const isTemplate = agentMeta?.is_template ? true : undefined;
        onSave(isTemplate);
    };

    const getType = () => {
        return agentMeta?.is_template ? 'template' : 'agent';
    }

    return (
        <Dialog>
            <Tooltip delayDuration={500}>
                <TooltipTrigger asChild>
                    <DialogTrigger asChild>
                        <Button
                            variant="ghost"
                            size="icon"
                        >
                            <Save />
                            <span className="sr-only">Save {getType()}</span>
                        </Button>
                    </DialogTrigger>
                </TooltipTrigger>
                <TooltipContent side="right">Save {getType()}</TooltipContent>
            </Tooltip>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle>Save</DialogTitle>
                    <DialogDescription>
                        Let&apos;s save this {getType()} so you can access it later.
                    </DialogDescription>
                </DialogHeader>
                <div className="grid gap-3 py-4">
                    <div>
                        <Label htmlFor="name">
                            Name
                        </Label>
                        <Input
                            id="name"
                            placeholder="Enter your agent name"
                            className="col-span-3"
                            defaultValue={agentMeta?.name || ''}
                            onChange={(e) => onNameChange(e.target.value)}
                        />
                    </div>
                    <div>
                        <Label htmlFor="description">
                            Description
                        </Label>
                        <Input
                            id="description"
                            placeholder="Your agent description"
                            className="col-span-3"
                            defaultValue={agentMeta?.description || ''}
                            onChange={(e) => onDescriptionChange(e.target.value)}
                        />
                    </div>
                </div>
                <DialogFooter>
                    <Button type="submit" onClick={handleSave}>Save {getType()}</Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}

export default SaveDialog;
