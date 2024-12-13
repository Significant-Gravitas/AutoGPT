// "use client";

// import {
//   Dialog,
//   DialogContent,
//   DialogClose,
//   DialogFooter,
//   DialogHeader,
//   DialogTitle,
//   DialogTrigger,
// } from "@/components/ui/dialog";
// import { Button } from "@/components/ui/button";
// import {
//   MultiSelector,
//   MultiSelectorContent,
//   MultiSelectorInput,
//   MultiSelectorItem,
//   MultiSelectorList,
//   MultiSelectorTrigger,
// } from "@/components/ui/multiselect";
// import { Controller, useForm } from "react-hook-form";
// import {
//   Select,
//   SelectContent,
//   SelectItem,
//   SelectTrigger,
//   SelectValue,
// } from "@/components/ui/select";
// import { useState } from "react";
// import { addFeaturedAgent } from "./actions";
// import { Agent } from "@/lib/marketplace-api/types";

// type FormData = {
//   agent: string;
//   categories: string[];
// };

// export const AdminAddFeaturedAgentDialog = ({
//   categories,
//   agents,
// }: {
//   categories: string[];
//   agents: Agent[];
// }) => {
//   const [selectedAgent, setSelectedAgent] = useState<string>("");
//   const [selectedCategories, setSelectedCategories] = useState<string[]>([]);

//   const {
//     control,
//     handleSubmit,
//     watch,
//     setValue,
//     formState: { errors },
//   } = useForm<FormData>({
//     defaultValues: {
//       agent: "",
//       categories: [],
//     },
//   });

//   return (
//     <Dialog>
//       <DialogTrigger asChild>
//         <Button variant="outline" size="sm">
//           Add Featured Agent
//         </Button>
//       </DialogTrigger>
//       <DialogContent>
//         <DialogHeader>
//           <DialogTitle>Add Featured Agent</DialogTitle>
//         </DialogHeader>
//         <div className="flex flex-col gap-4">
//           <Controller
//             name="agent"
//             control={control}
//             rules={{ required: true }}
//             render={({ field }) => (
//               <div>
//                 <label htmlFor={field.name}>Agent</label>
//                 <Select
//                   onValueChange={(value) => {
//                     field.onChange(value);
//                     setSelectedAgent(value);
//                   }}
//                   value={field.value || ""}
//                 >
//                   <SelectTrigger>
//                     <SelectValue placeholder="Select an agent" />
//                   </SelectTrigger>
//                   <SelectContent>
//                     {/* Populate with agents */}
//                     {agents.map((agent) => (
//                       <SelectItem key={agent.id} value={agent.id}>
//                         {agent.name}
//                       </SelectItem>
//                     ))}
//                   </SelectContent>
//                 </Select>
//               </div>
//             )}
//           />
//           <Controller
//             name="categories"
//             control={control}
//             render={({ field }) => (
//               <MultiSelector
//                 values={field.value || []}
//                 onValuesChange={(values) => {
//                   field.onChange(values);
//                   setSelectedCategories(values);
//                 }}
//               >
//                 <MultiSelectorTrigger>
//                   <MultiSelectorInput placeholder="Select categories" />
//                 </MultiSelectorTrigger>
//                 <MultiSelectorContent>
//                   <MultiSelectorList>
//                     {categories.map((category) => (
//                       <MultiSelectorItem key={category} value={category}>
//                         {category}
//                       </MultiSelectorItem>
//                     ))}
//                   </MultiSelectorList>
//                 </MultiSelectorContent>
//               </MultiSelector>
//             )}
//           />
//         </div>
//         <DialogFooter>
//           <DialogClose asChild>
//             <Button variant="outline">Cancel</Button>
//           </DialogClose>
//           <DialogClose asChild>
//             <Button
//               type="submit"
//               onClick={async () => {
//                 // Handle adding the featured agent
//                 await addFeaturedAgent(selectedAgent, selectedCategories);
//                 // close the dialog
//               }}
//             >
//               Add
//             </Button>
//           </DialogClose>
//         </DialogFooter>
//       </DialogContent>
//     </Dialog>
//   );
// };
