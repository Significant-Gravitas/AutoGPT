import { FC } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import useCredentials from "@/hooks/useCredentials";

interface ModalProps {

}

const CredentialsModal: FC<ModalProps> = ({

}) => {
  const credentials = useCredentials();

  if (credentials?.isLoading) {
    return <div>Loading...</div>;
  }

  if (!credentials) {
    return null;
  }

  const { isApiKey, schema } = credentials;

  return (
    {isApiKey &&
      <Select
        defaultValue={value}
        onValueChange={(newValue) => handleInputChange(selfKey, newValue)}
      >
        <SelectTrigger>
          <SelectValue placeholder={schema.placeholder || displayName} />
        </SelectTrigger>
        <SelectContent className="nodrag">
          {credentials.savedApiKeys.map((credentials, index) => (
            <SelectItem key={index} value={credentials.id}>
              {credentials.id} - {credentials.username}
            </SelectItem>
          ))}
          <SelectItem value="new">Add new API key...</SelectItem>
        </SelectContent>
      </Select>
    }
    
  );
};
