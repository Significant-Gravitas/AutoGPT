import { useMemo, useRef, useEffect } from "react";
import { CustomNodeData } from "../CustomNode";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useCustomNode } from "../useCustomNode";

type StickyNoteBlockType = {
 selected: boolean;
 data: CustomNodeData;
 nodeId: string;
};

export const StickyNoteBlock = ({ data, nodeId }: StickyNoteBlockType) => {
 const { angle, color } = useMemo(() => {
 const hash = nodeId.split("").reduce((acc, char) => {
 return char.charCodeAt(0) + ((acc << 5) - acc);
 }, 0);

 const colors = [
 "bg-orange-200",
 "bg-red-200",
 "bg-yellow-200",
 "bg-green-200",
 "bg-blue-200",
 "bg-purple-200",
 "bg-pink-200",
 ];

 return {
 angle: (hash % 7) - 3,
 color: colors[Math.abs(hash) % colors.length],
 };
 }, [nodeId]);

 const noteKey = Object.keys(data.inputSchema.properties || {}).find(
 (k) => k !== "output" && k !== "result",
 ) || "prompt";
 const noteContent = data.hardcodedValues?.[noteKey] ?? "";

 const { updateNodeData } = useCustomNode({ data, nodeId });
 const textareaRef = useRef<HTMLTextAreaElement>(null);

 const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
 const newValue = e.target.value;
 updateNodeData(nodeId, { [noteKey]: newValue });
 };

 useEffect(() => {
 const textarea = textareaRef.current;
 if (!textarea) return;
 const prev = textarea.value;
 const selStart = textarea.selectionStart;
 const selEnd = textarea.selectionEnd;
 if (prev !== noteContent) {
 textarea.value = noteContent;
 const len = noteContent.length;
 textarea.setSelectionRange(
 Math.min(selStart, len),
 Math.min(selEnd, len),
 );
 }
 });

 return (
 <div
 className={cn(
 "relative h-76 w-76 p-4 text-black shadow-[rgba(0,0,0,0.3)_-2px_5px_5px_0px]",
 color,
 )}
 style={{ transform: `rotate(${angle}deg)` }}
 >
 <Text variant="h3" className="tracking-tight text-slate-800">
 Notes #{nodeId.split("-")[0]}
 </Text>
 <textarea
 ref={textareaRef}
 value={noteContent}
 onChange={handleChange}
 placeholder="Write your note here..."
 className="!h-[230px] resize-none rounded-none border-none bg-transparent p-0 placeholder:text-black/60 focus:ring-0"
 />
 </div>
 );
};
