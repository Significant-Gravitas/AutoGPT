import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import useCredentials from "@/hooks/useCredentials";

interface ModalProps {

}

export default function CredentialsModal({ }: ModalProps) {
  const credentials = useCredentials();

  if (credentials?.isLoading) {
    return <div>Loading...</div>;
  }

  if (!credentials) {
    return null;
  }

  const { isApiKey, isOAuth2, schema, savedOAuthCredentials, oAuthLogin } = credentials;

  const handleValueChange = (newValue: string) => {
    if (newValue === "new") {
      // Trigger the action to add a new API key
      oAuthLogin("");//TODO kcze need scopes
    } else {
      // Handle normal selection
      //TODO kcze
    }
  };

  return (
    <div className="nodrag nowheel fixed inset-0 flex items-center justify-center bg-white bg-opacity-60">
      <div className="w-[500px] max-w-[90%] rounded-lg border-[1.5px] bg-white p-5">
        {isOAuth2 &&
          <Select
            defaultValue={schema.placeholder}
            onValueChange={(newValue) => { }}
          >
            <SelectTrigger>
              <SelectValue placeholder={schema.placeholder} />
            </SelectTrigger>
            <SelectContent className="nodrag">
              {savedOAuthCredentials.map((credentials, index) => (
                <SelectItem key={index} value={credentials.id}>
                  {credentials.id} - {credentials.username}
                </SelectItem>
              ))}
              <SelectItem value="new">Add new API key...</SelectItem>
            </SelectContent>
          </Select>
        }
      </div>
    </div>
  );
};
