import { useState, useEffect } from 'react';
import { CustomNodeData } from './types';

export const useCustomNode = (data: CustomNodeData, id: string) => {
  const [isPropertiesOpen, setIsPropertiesOpen] = useState(data.isPropertiesOpen || false);
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [modalValue, setModalValue] = useState<string>('');

  useEffect(() => {
    if (data.output_data || data.status) {
      setIsPropertiesOpen(true);
    }
  }, [data.output_data, data.status]);

  useEffect(() => {
    console.log(`Node ${id} data:`, data);
  }, [id, data]);

  const toggleProperties = () => setIsPropertiesOpen(!isPropertiesOpen);
  const toggleAdvancedSettings = () => setIsAdvancedOpen(!isAdvancedOpen);

  const handleInputClick = (key: string) => {
    setActiveKey(key);
    const value = getValue(key, data.hardcodedValues);
    setModalValue(typeof value === 'object' ? JSON.stringify(value, null, 2) : value);
    setIsModalOpen(true);
  };

  const handleModalSave = (value: string) => {
    if (activeKey) {
      try {
        const parsedValue = JSON.parse(value);
        data.setHardcodedValues({ ...data.hardcodedValues, [activeKey]: parsedValue });
      } catch (error) {
        data.setHardcodedValues({ ...data.hardcodedValues, [activeKey]: value });
      }
    }
    setIsModalOpen(false);
    setActiveKey(null);
  };

  return {
    isPropertiesOpen,
    isAdvancedOpen,
    isModalOpen,
    modalValue,
    activeKey,
    toggleProperties,
    toggleAdvancedSettings,
    handleInputClick,
    handleModalSave,
  };
};