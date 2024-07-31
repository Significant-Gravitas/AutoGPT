'use client'

import {
    Dialog,
    DialogContent,
    DialogDescription, DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger
} from "@/components/ui/dialog";
import {Button} from "@/components/ui/button";
import {Save} from "lucide-react";
import {Label} from "@/components/ui/label";
import {Input} from "@/components/ui/input";
import React from "react";



export const SaveDialog = () => {
    return (
        <Dialog>
            <DialogTrigger asChild>
                <Button
                    variant="ghost"
                    size="icon"
                >
                    <Save />
                    <span className="sr-only">Save</span>
                </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle>Save</DialogTitle>
                    <DialogDescription>
                        Save your agent to access it later. You can also save it as a template.
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
                        />
                    </div>
                </div>
                <DialogFooter>
                    <Button type="submit">Save Template</Button>
                    <Button type="submit">Save Agent</Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}

export default SaveDialog;