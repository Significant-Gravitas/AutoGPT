import React, { useState, useEffect, FC, memo } from 'react';
import { NodeProps } from 'reactflow';
import { CustomNodeData } from './types';
import { NodeHeader } from './NodeHeader';
import { NodeContent } from './NodeContent';
import { NodeProperties } from './NodeProperties';
import { NodeFooter } from './NodeFooter';
import { useCustomNode } from './hooks/useCustomNode';
import ModalComponent from '../ModalComponent';
import { hasOptionalFields } from "@/components/CustomNode/utils";

const CustomNode: FC<NodeProps<CustomNodeData>> = ({ data, id }) => {
  const {
    isPropertiesOpen,
    isAdvancedOpen,
    isModalOpen,
    modalValue,
    activeKey,
    toggleProperties,
    toggleAdvancedSettings,
    handleInputClick,
    handleModalSave,
    setIsModalOpen,
  } = useCustomNode(data, id);

  return (
    <div className={`custom-node dark-theme ${data.status === 'RUNNING' ? 'running' : data.status === 'COMPLETED' ? 'completed' : data.status === 'FAILED' ? 'failed' : ''}`}>
      <NodeHeader title={data.blockType || data.title} />
      <NodeContent
        data={data}
        isAdvancedOpen={isAdvancedOpen}
        handleInputClick={handleInputClick}
      />
      {isPropertiesOpen && <NodeProperties status={data.status} outputData={data.output_data} />}
      <NodeFooter
        toggleProperties={toggleProperties}
        toggleAdvancedSettings={toggleAdvancedSettings}
        hasOptionalFields={hasOptionalFields(data.inputSchema)}
      />
      <ModalComponent
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleModalSave}
        value={modalValue}
      />
    </div>
  );
};

export default memo(CustomNode);