import React, { FC } from "react";
import { Button } from "./ui/button";
import { ContentRenderer } from "./ui/render";
import { beautifyString } from "@/lib/utils";
import { Clipboard, Maximize2 } from "lucide-react";
import { useToast } from "./molecules/Toast/use-toast";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "./ui/dialog";
import { ScrollArea } from "./ui/scroll-area";
import { Separator } from "./ui/separator";

interface ExpandableOutputDialogProps {
  isOpen: boolean;
  onClose: () => void;
  execId: string;
  pinName: string;
  data: any[];
}

const ExpandableOutputDialog: FC<ExpandableOutputDialogProps> = ({
  isOpen,
  onClose,
  execId,
  pinName,
  data,
}) => {
  const { toast } = useToast();

  const copyData = () => {
    const formattedData = data
      .map((item) =>
        typeof item === "object" ? JSON.stringify(item, null, 2) : String(item),
      )
      .join("\n\n");
    
    navigator.clipboard.writeText(formattedData).then(() => {
      toast({
        title: `"${beautifyString(pinName)}" data copied to clipboard!`,
        duration: 2000,
      });
    });
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl w-[90vw] h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Maximize2 size={20} />
            Full Output Preview
          </DialogTitle>
          <DialogDescription>
            Execution ID: <span className="font-mono text-xs">{execId}</span>
            <br />
            Pin: <span className="font-semibold">{beautifyString(pinName)}</span>
          </DialogDescription>
        </DialogHeader>
        
        <div className="flex-1 overflow-hidden">
          <ScrollArea className="h-full">
            <div className="p-4">
              {data.length > 0 ? (
                <div className="space-y-4">
                  {data.map((item, index) => (
                    <div key={index} className="border rounded-lg p-4 bg-gray-50">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-600">
                          Item {index + 1} of {data.length}
                        </span>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            const itemData = typeof item === "object" 
                              ? JSON.stringify(item, null, 2) 
                              : String(item);
                            navigator.clipboard.writeText(itemData).then(() => {
                              toast({
                                title: `Item ${index + 1} copied to clipboard!`,
                                duration: 2000,
                              });
                            });
                          }}
                          className="flex items-center gap-1"
                        >
                          <Clipboard size={14} />
                          Copy Item
                        </Button>
                      </div>
                      <Separator className="mb-3" />
                      <div className="whitespace-pre-wrap break-words font-mono text-sm">
                        <ContentRenderer 
                          value={item} 
                          truncateLongData={false} 
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-gray-500 py-8">
                  No data available
                </div>
              )}
            </div>
          </ScrollArea>
        </div>

        <DialogFooter className="flex justify-between">
          <div className="text-sm text-gray-600">
            {data.length} item{data.length !== 1 ? 's' : ''} total
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={copyData}
              className="flex items-center gap-1"
            >
              <Clipboard size={16} />
              Copy All
            </Button>
            <Button onClick={onClose}>Close</Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default ExpandableOutputDialog;