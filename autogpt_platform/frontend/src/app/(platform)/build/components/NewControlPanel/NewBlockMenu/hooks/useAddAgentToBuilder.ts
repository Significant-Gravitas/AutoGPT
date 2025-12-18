import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useShallow } from "zustand/react/shallow";
import { useReactFlow } from "@xyflow/react";
import { convertLibraryAgentIntoCustomNode } from "../helpers";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { getV2GetLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";

export const useAddAgentToBuilder = () => {
  const addBlock = useNodeStore(useShallow((state) => state.addBlock));
  const { setViewport } = useReactFlow();

  const addAgentToBuilder = (libraryAgent: LibraryAgent) => {
    const { input_schema, output_schema } = libraryAgent;

    const { block, hardcodedValues } = convertLibraryAgentIntoCustomNode(
      libraryAgent,
      input_schema,
      output_schema,
    );

    const customNode = addBlock(block, hardcodedValues);

    setTimeout(() => {
      setViewport(
        {
          x: -customNode.position.x * 0.8 + window.innerWidth / 2,
          y: -customNode.position.y * 0.8 + (window.innerHeight - 400) / 2,
          zoom: 0.8,
        },
        { duration: 500 },
      );
    }, 50);

    return customNode;
  };

  const addLibraryAgentToBuilder = async (agent: LibraryAgent) => {
    const response = await getV2GetLibraryAgent(agent.id);

    if (!response.data) {
      throw new Error("Failed to get agent details");
    }

    const libraryAgent = response.data as LibraryAgent;
    return addAgentToBuilder(libraryAgent);
  };

  return {
    addAgentToBuilder,
    addLibraryAgentToBuilder,
  };
};
