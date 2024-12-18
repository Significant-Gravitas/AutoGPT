import { FaSpinner } from "react-icons/fa";

export default function Spinner() {
  return (
    <div className="flex h-[80vh] items-center justify-center">
      <FaSpinner className="mr-2 h-16 w-16 animate-spin" />
    </div>
  );
}
