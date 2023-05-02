import React, { useState } from "react";
import { useTranslation } from "next-i18next";
import type { Language } from "../utils/languages";
import { ENGLISH, languages } from "../utils/languages";
import { useRouter } from "next/router";
import Input from "./Input";
import { FaGlobe } from "react-icons/fa";

const LanguageCombobox = () => {
  const { i18n } = useTranslation();
  const router = useRouter();
  const [actualLanguage, setActualLanguage] = useState(
    findLanguage(i18n.language)
  );

  const handleInputChange = (languageName: string) => {
    const selectedLanguage = findLanguage(languageName);
    setActualLanguage(selectedLanguage);
    handleLanguageChange(selectedLanguage.code);
  };

  const handleLanguageChange = (value: string) => {
    const { pathname, asPath, query } = router;
    router
      .push({ pathname, query }, asPath, {
        locale: value,
      })
      .catch(console.error);
  };

  return (
    <Input
      left={
        <>
          <FaGlobe />
          <span className="ml-2">Lang:</span>
        </>
      }
      type="combobox"
      value={actualLanguage.name}
      onChange={(e) => handleInputChange(e.target.value)}
      setValue={(e) => handleInputChange(e)}
      attributes={{ options: languages.map((lang) => lang.name) }}
    />
  );
};

const findLanguage = (nameOrLocale: string): Language => {
  const selectedLanguage = languages.find(
    (lang) => lang.code === nameOrLocale || lang.name === nameOrLocale
  );
  return selectedLanguage || ENGLISH;
};

export default LanguageCombobox;
