import React from 'react';
import { beautifyString } from '@/lib/utils';

interface NodeHeaderProps {
  title: string;
}

export const NodeHeader: React.FC<NodeHeaderProps> = ({ title }) => {
  return (
    <div className="node-header">
      <div className="node-title">{beautifyString(title.replace(/Block$/, ''))}</div>
    </div>
  );
};