import { Collapsible } from "@radix-ui/react-collapsible";
import {CollapsibleContent, CollapsibleTrigger} from "@/components/ui/collapsible";
import {Save} from "lucide-react";
import {Button} from "@/components/ui/button";
import React from "react";

interface SaveControlProps {

}

export const SaveControl= ( props:  SaveControlProps) => {

    return (
        <Collapsible>
            <CollapsibleTrigger>
                <Button
                    variant="ghost"
                    size="icon"
                >
                    <Save className={"size-5"}/>
                    <span className="sr-only">Save</span>
                </Button>
                </CollapsibleTrigger>
            <CollapsibleContent>
                Yes. Free to use for personal and commercial projects. No attribution
                required.
            </CollapsibleContent>
        </Collapsible>
    );
}