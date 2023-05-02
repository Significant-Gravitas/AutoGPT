import { Fragment, memo } from "react";
import { Menu as MenuPrimitive } from "@headlessui/react";
import { FaChevronDown } from "react-icons/fa";

interface MenuProps {
  name: string;
  items: JSX.Element[];
  disabled?: boolean;
  onChange: (value: string) => void;
  styleClass?: { [key: string]: string };
}

function Menu({ name, items, disabled, onChange, styleClass }: MenuProps) {
  return (
    <MenuPrimitive>
      <div className={styleClass?.container}>
        <MenuPrimitive.Button className={styleClass?.input}>
          <span>{name}</span>
          <FaChevronDown
            className="absolute right-1.5 inline-block h-5 w-5 text-gray-400"
            aria-hidden="true"
          />
        </MenuPrimitive.Button>
        <MenuPrimitive.Items className="absolute right-0 top-full z-20 mt-1 max-h-48 w-full overflow-hidden rounded-xl border-[2px] border-white/10 bg-[#3a3a3a] tracking-wider shadow-xl outline-0 transition-all">
          {items.map((item) => {
            const itemName = (item.props as { name: string }).name;
            return (
              <MenuPrimitive.Item key={itemName} as={Fragment}>
                <div className={styleClass?.option}>{item}</div>
              </MenuPrimitive.Item>
            );
          })}
        </MenuPrimitive.Items>
      </div>
    </MenuPrimitive>
  );
}

export default memo(Menu);
