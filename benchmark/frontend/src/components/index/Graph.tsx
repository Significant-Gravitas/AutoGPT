import React, { useEffect, useRef, useState } from "react";
import { Network } from "vis-network";
import { DataSet } from "vis-data";

import tw from "tailwind-styled-components";

import { GraphNode, TaskData } from "../../lib/types";

interface GraphEdge {
  id: string;
  from: string;
  to: string;
  arrows: string;
}

interface GraphProps {
  graphData: {
    nodes: GraphNode[];
    edges: GraphEdge[];
  };
  setSelectedTask: React.Dispatch<React.SetStateAction<TaskData | null>>;
  setIsTaskInfoExpanded: React.Dispatch<React.SetStateAction<boolean>>;
}

const Graph: React.FC<GraphProps> = ({
  graphData,
  setSelectedTask,
  setIsTaskInfoExpanded,
}) => {
  const graphRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!graphRef.current) {
      return;
    }
    const nodes = new DataSet<GraphNode>(graphData.nodes);
    const edges = new DataSet<GraphEdge>(graphData.edges);

    const data = {
      nodes: nodes,
      edges: edges,
    };

    const options = {
      nodes: {
        font: {
          size: 20, // Increased font size for labels
          color: "black", // Set a readable font color
        },
        shapeProperties: {
          useBorderWithImage: true,
        },
      },
      edges: {
        length: 250, // Increased edge length
      },
      layout: {
        hierarchical: {
          enabled: true,
          levelSeparation: 300,
          nodeSpacing: 250,
          treeSpacing: 250,
          blockShifting: true,
          edgeMinimization: true,
          parentCentralization: true,
          direction: "UD",
          sortMethod: "directed",
        },
      },
      physics: {
        stabilization: {
          enabled: true,
          iterations: 1000,
        },
        hierarchicalRepulsion: {
          centralGravity: 0.0,
          springLength: 200,
          springConstant: 0.01,
          nodeDistance: 300,
          damping: 0.09,
        },
        timestep: 0.5,
      },
    };

    const network = new Network(graphRef.current, data, options);

    // Add an event listener for node clicks
    network.on("click", (params) => {
      if (params.nodes.length) {
        const nodeId = params.nodes[0];
        const clickedNodeArray = nodes.get(nodeId);
        if (clickedNodeArray) {
          setSelectedTask((clickedNodeArray as any).data as TaskData);
          setIsTaskInfoExpanded(true);
        }
      } else {
        setSelectedTask(null);
        setIsTaskInfoExpanded(false);
      }
    });
  }, [graphData]);

  return <GraphContainer ref={graphRef} />;
};

export default Graph;

const GraphContainer = tw.div`
  w-full
  h-full
`;
