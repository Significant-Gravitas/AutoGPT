import { BlockSchema } from "@/lib/types";
import { beautifyString, getTypeBgColor, getTypeTextColor } from "@/lib/utils";
import { FC } from "react";
import { Handle, Position } from "reactflow";
import SchemaTooltip from "./SchemaTooltip";

type HandleProps = {
  keyName: string,
  schema: BlockSchema,
  isConnected: boolean,
  isRequired?: boolean,
  side: 'left' | 'right'
}

const NodeHandle: FC<HandleProps> = ({ keyName, schema, isConnected, isRequired, side }) => {

  const typeName: Record<string, string> = {
    string: 'text',
    number: 'number',
    boolean: 'true/false',
    object: 'complex',
    array: 'list',
    null: 'null',
  };

  const typeClass = `text-sm ${getTypeTextColor(schema.type)} ${side === 'left' ? 'text-left' : 'text-right'}`;

  const label = (
    <div className="flex flex-col flex-grow">
      <span className="text-m text-gray-900 -mb-1 green">
        {schema.title || beautifyString(keyName)}{isRequired ? '*' : ''}
      </span>
      <span className={typeClass}>{typeName[schema.type]}</span>
    </div>
  );

  const dot = (
    <div className={`w-4 h-4 m-1 ${isConnected ? getTypeBgColor(schema.type) : 'bg-gray-600'} rounded-full transition-colors duration-100 group-hover:bg-gray-300`} />
  );

  if (side === 'left') {
    return (
      <div key={keyName} className="handle-container">
        <Handle
          type="target"
          position={Position.Left}
          id={keyName}
          className='group -ml-[29px]'
        >
          <div className="pointer-events-none flex items-center">
            {dot}
            {label}
          </div>
        </Handle>
        <SchemaTooltip schema={schema} />
      </div>
    )
  } else {
    return (
      <div key={keyName} className="handle-container justify-end">
        <Handle
          type="source"
          position={Position.Right}
          id={keyName}
          className='group -mr-[29px]'
        >
          <div className="pointer-events-none flex items-center">
            {label}
            {dot}
          </div>
        </Handle>
      </div >
    )
  }
}

export default NodeHandle;
