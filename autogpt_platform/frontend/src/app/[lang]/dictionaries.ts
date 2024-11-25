import "server-only";

const dictionaries: Record<string, () => Promise<any>> = {
  en: () => import("./dictionaries/en.json").then((module) => module.default),
  es: () => import("./dictionaries/es.json").then((module) => module.default),
};

export const getDictionary = async (locale: string): Promise<any> => {
  const localeKey = locale || "en";
  if (!dictionaries[localeKey]) {
    return dictionaries["en"]();
  }
  return dictionaries[localeKey]();
};
