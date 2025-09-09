import base64
from enum import Enum
from logging import getLogger
from typing import Any
from urllib.parse import quote, urlencode

from backend.sdk import BaseModel, Credentials, Requests

logger = getLogger(__name__)


def _convert_bools(
    obj: Any,
) -> Any:  # noqa: ANN401 – allow Any for deep conversion utility
    """Recursively walk *obj* and coerce string booleans to real booleans."""
    if isinstance(obj, str):
        lowered = obj.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return obj
    if isinstance(obj, list):
        return [_convert_bools(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _convert_bools(v) for k, v in obj.items()}
    return obj


class WebhookFilters(BaseModel):
    dataTypes: list[str]
    changeTypes: list[str] | None = None
    fromSources: list[str] | None = None
    sourceOptions: dict | None = None
    watchDataInFieldIds: list[str] | None = None
    watchSchemasOfFieldIds: list[str] | None = None


class WebhookIncludes(BaseModel):
    includeCellValuesInFieldIds: list[str] | str | None = None
    includePreviousCellValues: bool | None = None
    includePreviousFieldDefinitions: bool | None = None


class WebhookSpecification(BaseModel):
    recordChangeScope: str | None = None
    filters: WebhookFilters
    includes: WebhookIncludes | None = None


class WebhookPayload(BaseModel):
    actionMetadata: dict
    baseTransactionNumber: int
    payloadFormat: str
    timestamp: str
    changedTablesById: dict | None = None
    createdTablesById: dict | None = None
    destroyedTableIds: list[str] | None = None
    error: bool | None = None
    code: str | None = None


class ListWebhookPayloadsResponse(BaseModel):
    payloads: list[WebhookPayload]
    cursor: int | None = None
    might_have_more: bool | None = None
    payloadFormat: str


class TableFieldType(str, Enum):
    SINGLE_LINE_TEXT = "singleLineText"
    EMAIL = "email"
    URL = "url"
    MULTILINE_TEXT = "multilineText"
    NUMBER = "number"
    PERCENT = "percent"
    CURRENCY = "currency"
    SINGLE_SELECT = "singleSelect"
    MULTIPLE_SELECTS = "multipleSelects"
    SINGLE_COLLABORATOR = "singleCollaborator"
    MULTIPLE_COLLABORATORS = "multipleCollaborators"
    MULTIPLE_RECORD_LINKS = "multipleRecordLinks"
    DATE = "date"
    DATE_TIME = "dateTime"
    PHONE_NUMBER = "phoneNumber"
    MULTIPLE_ATTACHMENTS = "multipleAttachments"
    CHECKBOX = "checkbox"
    FORMULA = "formula"
    CREATED_TIME = "createdTime"
    ROLLUP = "rollup"
    COUNT = "count"
    LOOKUP = "lookup"
    MULTIPLE_LOOKUP_VALUES = "multipleLookupValues"
    AUTO_NUMBER = "autoNumber"
    BARCODE = "barcode"
    RATING = "rating"
    RICH_TEXT = "richText"
    DURATION = "duration"
    LAST_MODIFIED_TIME = "lastModifiedTime"
    BUTTON = "button"
    CREATED_BY = "createdBy"
    LAST_MODIFIED_BY = "lastModifiedBy"
    EXTERNAL_SYNC_SOURCE = "externalSyncSource"
    AI_TEXT = "aiText"


TABLE_FIELD_TYPES = set(type.value for type in TableFieldType)


class AirtableTimeZones(str, Enum):
    UTC = "utc"
    CLIENT = "client"
    AFRICA_ABIDJAN = "Africa/Abidjan"
    AFRICA_ACCRA = "Africa/Accra"
    AFRICA_ADDIS_ABABA = "Africa/Addis_Ababa"
    AFRICA_ALGIERS = "Africa/Algiers"
    AFRICA_ASMARA = "Africa/Asmara"
    AFRICA_BAMAKO = "Africa/Bamako"
    AFRICA_BANGUI = "Africa/Bangui"
    AFRICA_BANJUL = "Africa/Banjul"
    AFRICA_BISSAU = "Africa/Bissau"
    AFRICA_BLANTYRE = "Africa/Blantyre"
    AFRICA_BRAZZAVILLE = "Africa/Brazzaville"
    AFRICA_BUJUMBURA = "Africa/Bujumbura"
    AFRICA_CAIRO = "Africa/Cairo"
    AFRICA_CASABLANCA = "Africa/Casablanca"
    AFRICA_CEUTA = "Africa/Ceuta"
    AFRICA_CONAKRY = "Africa/Conakry"
    AFRICA_DAKAR = "Africa/Dakar"
    AFRICA_DAR_ES_SALAAM = "Africa/Dar_es_Salaam"
    AFRICA_DJIBOUTI = "Africa/Djibouti"
    AFRICA_DOUALA = "Africa/Douala"
    AFRICA_EL_AAIUN = "Africa/El_Aaiun"
    AFRICA_FREETOWN = "Africa/Freetown"
    AFRICA_GABORONE = "Africa/Gaborone"
    AFRICA_HARARE = "Africa/Harare"
    AFRICA_JOHANNESBURG = "Africa/Johannesburg"
    AFRICA_JUBA = "Africa/Juba"
    AFRICA_KAMPALA = "Africa/Kampala"
    AFRICA_KHARTOUM = "Africa/Khartoum"
    AFRICA_KIGALI = "Africa/Kigali"
    AFRICA_KINSHASA = "Africa/Kinshasa"
    AFRICA_LAGOS = "Africa/Lagos"
    AFRICA_LIBREVILLE = "Africa/Libreville"
    AFRICA_LOME = "Africa/Lome"
    AFRICA_LUANDA = "Africa/Luanda"
    AFRICA_LUBUMBASHI = "Africa/Lubumbashi"
    AFRICA_LUSAKA = "Africa/Lusaka"
    AFRICA_MALABO = "Africa/Malabo"
    AFRICA_MAPUTO = "Africa/Maputo"
    AFRICA_MASERU = "Africa/Maseru"
    AFRICA_MBABANE = "Africa/Mbabane"
    AFRICA_MOGADISHU = "Africa/Mogadishu"
    AFRICA_MONROVIA = "Africa/Monrovia"
    AFRICA_NAIROBI = "Africa/Nairobi"
    AFRICA_NDJAMENA = "Africa/Ndjamena"
    AFRICA_NIAMEY = "Africa/Niamey"
    AFRICA_NOUAKCHOTT = "Africa/Nouakchott"
    AFRICA_OUAGADOUGOU = "Africa/Ouagadougou"
    AFRICA_PORTO_NOVO = "Africa/Porto-Novo"
    AFRICA_SAO_TOME = "Africa/Sao_Tome"
    AFRICA_TRIPOLI = "Africa/Tripoli"
    AFRICA_TUNIS = "Africa/Tunis"
    AFRICA_WINDHOEK = "Africa/Windhoek"
    AMERICA_ADAK = "America/Adak"
    AMERICA_ANCHORAGE = "America/Anchorage"
    AMERICA_ANGUILLA = "America/Anguilla"
    AMERICA_ANTIGUA = "America/Antigua"
    AMERICA_ARAGUAINA = "America/Araguaina"
    AMERICA_ARGENTINA_BUENOS_AIRES = "America/Argentina/Buenos_Aires"
    AMERICA_ARGENTINA_CATAMARCA = "America/Argentina/Catamarca"
    AMERICA_ARGENTINA_CORDOBA = "America/Argentina/Cordoba"
    AMERICA_ARGENTINA_JUJUY = "America/Argentina/Jujuy"
    AMERICA_ARGENTINA_LA_RIOJA = "America/Argentina/La_Rioja"
    AMERICA_ARGENTINA_MENDOZA = "America/Argentina/Mendoza"
    AMERICA_ARGENTINA_RIO_GALLEGOS = "America/Argentina/Rio_Gallegos"
    AMERICA_ARGENTINA_SALTA = "America/Argentina/Salta"
    AMERICA_ARGENTINA_SAN_JUAN = "America/Argentina/San_Juan"
    AMERICA_ARGENTINA_SAN_LUIS = "America/Argentina/San_Luis"
    AMERICA_ARGENTINA_TUCUMAN = "America/Argentina/Tucuman"
    AMERICA_ARGENTINA_USHUAIA = "America/Argentina/Ushuaia"
    AMERICA_ARUBA = "America/Aruba"
    AMERICA_ASUNCION = "America/Asuncion"
    AMERICA_ATIKOKAN = "America/Atikokan"
    AMERICA_BAHIA = "America/Bahia"
    AMERICA_BAHIA_BANDERAS = "America/Bahia_Banderas"
    AMERICA_BARBADOS = "America/Barbados"
    AMERICA_BELEM = "America/Belem"
    AMERICA_BELIZE = "America/Belize"
    AMERICA_BLANC_SABLON = "America/Blanc-Sablon"
    AMERICA_BOA_VISTA = "America/Boa_Vista"
    AMERICA_BOGOTA = "America/Bogota"
    AMERICA_BOISE = "America/Boise"
    AMERICA_CAMBRIDGE_BAY = "America/Cambridge_Bay"
    AMERICA_CAMPO_GRANDE = "America/Campo_Grande"
    AMERICA_CANCUN = "America/Cancun"
    AMERICA_CARACAS = "America/Caracas"
    AMERICA_CAYENNE = "America/Cayenne"
    AMERICA_CAYMAN = "America/Cayman"
    AMERICA_CHICAGO = "America/Chicago"
    AMERICA_CHIHUAHUA = "America/Chihuahua"
    AMERICA_COSTA_RICA = "America/Costa_Rica"
    AMERICA_CRESTON = "America/Creston"
    AMERICA_CUIABA = "America/Cuiaba"
    AMERICA_CURACAO = "America/Curacao"
    AMERICA_DANMARKSHAVN = "America/Danmarkshavn"
    AMERICA_DAWSON = "America/Dawson"
    AMERICA_DAWSON_CREEK = "America/Dawson_Creek"
    AMERICA_DENVER = "America/Denver"
    AMERICA_DETROIT = "America/Detroit"
    AMERICA_DOMINICA = "America/Dominica"
    AMERICA_EDMONTON = "America/Edmonton"
    AMERICA_EIRUNEPE = "America/Eirunepe"
    AMERICA_EL_SALVADOR = "America/El_Salvador"
    AMERICA_FORT_NELSON = "America/Fort_Nelson"
    AMERICA_FORTALEZA = "America/Fortaleza"
    AMERICA_GLACE_BAY = "America/Glace_Bay"
    AMERICA_GODTHAB = "America/Godthab"
    AMERICA_GOOSE_BAY = "America/Goose_Bay"
    AMERICA_GRAND_TURK = "America/Grand_Turk"
    AMERICA_GRENADA = "America/Grenada"
    AMERICA_GUADELOUPE = "America/Guadeloupe"
    AMERICA_GUATEMALA = "America/Guatemala"
    AMERICA_GUAYAQUIL = "America/Guayaquil"
    AMERICA_GUYANA = "America/Guyana"
    AMERICA_HALIFAX = "America/Halifax"
    AMERICA_HAVANA = "America/Havana"
    AMERICA_HERMOSILLO = "America/Hermosillo"
    AMERICA_INDIANA_INDIANAPOLIS = "America/Indiana/Indianapolis"
    AMERICA_INDIANA_KNOX = "America/Indiana/Knox"
    AMERICA_INDIANA_MARENGO = "America/Indiana/Marengo"
    AMERICA_INDIANA_PETERSBURG = "America/Indiana/Petersburg"
    AMERICA_INDIANA_TELL_CITY = "America/Indiana/Tell_City"
    AMERICA_INDIANA_VEVAY = "America/Indiana/Vevay"
    AMERICA_INDIANA_VINCENNES = "America/Indiana/Vincennes"
    AMERICA_INDIANA_WINAMAC = "America/Indiana/Winamac"
    AMERICA_INUVIK = "America/Inuvik"
    AMERICA_IQALUIT = "America/Iqaluit"
    AMERICA_JAMAICA = "America/Jamaica"
    AMERICA_JUNEAU = "America/Juneau"
    AMERICA_KENTUCKY_LOUISVILLE = "America/Kentucky/Louisville"
    AMERICA_KENTUCKY_MONTICELLO = "America/Kentucky/Monticello"
    AMERICA_KRALENDIJK = "America/Kralendijk"
    AMERICA_LA_PAZ = "America/La_Paz"
    AMERICA_LIMA = "America/Lima"
    AMERICA_LOS_ANGELES = "America/Los_Angeles"
    AMERICA_LOWER_PRINCES = "America/Lower_Princes"
    AMERICA_MACEIO = "America/Maceio"
    AMERICA_MANAGUA = "America/Managua"
    AMERICA_MANAUS = "America/Manaus"
    AMERICA_MARIGOT = "America/Marigot"
    AMERICA_MARTINIQUE = "America/Martinique"
    AMERICA_MATAMOROS = "America/Matamoros"
    AMERICA_MAZATLAN = "America/Mazatlan"
    AMERICA_MENOMINEE = "America/Menominee"
    AMERICA_MERIDA = "America/Merida"
    AMERICA_METLAKATLA = "America/Metlakatla"
    AMERICA_MEXICO_CITY = "America/Mexico_City"
    AMERICA_MIQUELON = "America/Miquelon"
    AMERICA_MONCTON = "America/Moncton"
    AMERICA_MONTERREY = "America/Monterrey"
    AMERICA_MONTEVIDEO = "America/Montevideo"
    AMERICA_MONTSERRAT = "America/Montserrat"
    AMERICA_NASSAU = "America/Nassau"
    AMERICA_NEW_YORK = "America/New_York"
    AMERICA_NIPIGON = "America/Nipigon"
    AMERICA_NOME = "America/Nome"
    AMERICA_NORONHA = "America/Noronha"
    AMERICA_NORTH_DAKOTA_BEULAH = "America/North_Dakota/Beulah"
    AMERICA_NORTH_DAKOTA_CENTER = "America/North_Dakota/Center"
    AMERICA_NORTH_DAKOTA_NEW_SALEM = "America/North_Dakota/New_Salem"
    AMERICA_NUUK = "America/Nuuk"
    AMERICA_OJINAGA = "America/Ojinaga"
    AMERICA_PANAMA = "America/Panama"
    AMERICA_PANGNIRTUNG = "America/Pangnirtung"
    AMERICA_PARAMARIBO = "America/Paramaribo"
    AMERICA_PHOENIX = "America/Phoenix"
    AMERICA_PORT_AU_PRINCE = "America/Port-au-Prince"
    AMERICA_PORT_OF_SPAIN = "America/Port_of_Spain"
    AMERICA_PORTO_VELHO = "America/Porto_Velho"
    AMERICA_PUERTO_RICO = "America/Puerto_Rico"
    AMERICA_PUNTA_ARENAS = "America/Punta_Arenas"
    AMERICA_RAINY_RIVER = "America/Rainy_River"
    AMERICA_RANKIN_INLET = "America/Rankin_Inlet"
    AMERICA_RECIFE = "America/Recife"
    AMERICA_REGINA = "America/Regina"
    AMERICA_RESOLUTE = "America/Resolute"
    AMERICA_RIO_BRANCO = "America/Rio_Branco"
    AMERICA_SANTAREM = "America/Santarem"
    AMERICA_SANTIAGO = "America/Santiago"
    AMERICA_SANTO_DOMINGO = "America/Santo_Domingo"
    AMERICA_SAO_PAULO = "America/Sao_Paulo"
    AMERICA_SCORESBYSUND = "America/Scoresbysund"
    AMERICA_SITKA = "America/Sitka"
    AMERICA_ST_BARTHELEMY = "America/St_Barthelemy"
    AMERICA_ST_JOHNS = "America/St_Johns"
    AMERICA_ST_KITTS = "America/St_Kitts"
    AMERICA_ST_LUCIA = "America/St_Lucia"
    AMERICA_ST_THOMAS = "America/St_Thomas"
    AMERICA_ST_VINCENT = "America/St_Vincent"
    AMERICA_SWIFT_CURRENT = "America/Swift_Current"
    AMERICA_TEGUCIGALPA = "America/Tegucigalpa"
    AMERICA_THULE = "America/Thule"
    AMERICA_THUNDER_BAY = "America/Thunder_Bay"
    AMERICA_TIJUANA = "America/Tijuana"
    AMERICA_TORONTO = "America/Toronto"
    AMERICA_TORTOLA = "America/Tortola"
    AMERICA_VANCOUVER = "America/Vancouver"
    AMERICA_WHITEHORSE = "America/Whitehorse"
    AMERICA_WINNIPEG = "America/Winnipeg"
    AMERICA_YAKUTAT = "America/Yakutat"
    AMERICA_YELLOWKNIFE = "America/Yellowknife"
    ANTARCTICA_CASEY = "Antarctica/Casey"
    ANTARCTICA_DAVIS = "Antarctica/Davis"
    ANTARCTICA_DUMONT_DURVILLE = "Antarctica/DumontDUrville"
    ANTARCTICA_MACQUARIE = "Antarctica/Macquarie"
    ANTARCTICA_MAWSON = "Antarctica/Mawson"
    ANTARCTICA_MCMURDO = "Antarctica/McMurdo"
    ANTARCTICA_PALMER = "Antarctica/Palmer"
    ANTARCTICA_ROTHERA = "Antarctica/Rothera"
    ANTARCTICA_SYOWA = "Antarctica/Syowa"
    ANTARCTICA_TROLL = "Antarctica/Troll"
    ANTARCTICA_VOSTOK = "Antarctica/Vostok"
    ARCTIC_LONGYEARBYEN = "Arctic/Longyearbyen"
    ASIA_ADEN = "Asia/Aden"
    ASIA_ALMATY = "Asia/Almaty"
    ASIA_AMMAN = "Asia/Amman"
    ASIA_ANADYR = "Asia/Anadyr"
    ASIA_AQTAU = "Asia/Aqtau"
    ASIA_AQTOBE = "Asia/Aqtobe"
    ASIA_ASHGABAT = "Asia/Ashgabat"
    ASIA_ATYRAU = "Asia/Atyrau"
    ASIA_BAGHDAD = "Asia/Baghdad"
    ASIA_BAHRAIN = "Asia/Bahrain"
    ASIA_BAKU = "Asia/Baku"
    ASIA_BANGKOK = "Asia/Bangkok"
    ASIA_BARNAUL = "Asia/Barnaul"
    ASIA_BEIRUT = "Asia/Beirut"
    ASIA_BISHKEK = "Asia/Bishkek"
    ASIA_BRUNEI = "Asia/Brunei"
    ASIA_CHITA = "Asia/Chita"
    ASIA_CHOIBALSAN = "Asia/Choibalsan"
    ASIA_COLOMBO = "Asia/Colombo"
    ASIA_DAMASCUS = "Asia/Damascus"
    ASIA_DHAKA = "Asia/Dhaka"
    ASIA_DILI = "Asia/Dili"
    ASIA_DUBAI = "Asia/Dubai"
    ASIA_DUSHANBE = "Asia/Dushanbe"
    ASIA_FAMAGUSTA = "Asia/Famagusta"
    ASIA_GAZA = "Asia/Gaza"
    ASIA_HEBRON = "Asia/Hebron"
    ASIA_HO_CHI_MINH = "Asia/Ho_Chi_Minh"
    ASIA_HONG_KONG = "Asia/Hong_Kong"
    ASIA_HOVD = "Asia/Hovd"
    ASIA_IRKUTSK = "Asia/Irkutsk"
    ASIA_ISTANBUL = "Asia/Istanbul"
    ASIA_JAKARTA = "Asia/Jakarta"
    ASIA_JAYAPURA = "Asia/Jayapura"
    ASIA_JERUSALEM = "Asia/Jerusalem"
    ASIA_KABUL = "Asia/Kabul"
    ASIA_KAMCHATKA = "Asia/Kamchatka"
    ASIA_KARACHI = "Asia/Karachi"
    ASIA_KATHMANDU = "Asia/Kathmandu"
    ASIA_KHANDYGA = "Asia/Khandyga"
    ASIA_KOLKATA = "Asia/Kolkata"
    ASIA_KRASNOYARSK = "Asia/Krasnoyarsk"
    ASIA_KUALA_LUMPUR = "Asia/Kuala_Lumpur"
    ASIA_KUCHING = "Asia/Kuching"
    ASIA_KUWAIT = "Asia/Kuwait"
    ASIA_MACAU = "Asia/Macau"
    ASIA_MAGADAN = "Asia/Magadan"
    ASIA_MAKASSAR = "Asia/Makassar"
    ASIA_MANILA = "Asia/Manila"
    ASIA_MUSCAT = "Asia/Muscat"
    ASIA_NICOSIA = "Asia/Nicosia"
    ASIA_NOVOKUZNETSK = "Asia/Novokuznetsk"
    ASIA_NOVOSIBIRSK = "Asia/Novosibirsk"
    ASIA_OMSK = "Asia/Omsk"
    ASIA_ORAL = "Asia/Oral"
    ASIA_PHNOM_PENH = "Asia/Phnom_Penh"
    ASIA_PONTIANAK = "Asia/Pontianak"
    ASIA_PYONGYANG = "Asia/Pyongyang"
    ASIA_QATAR = "Asia/Qatar"
    ASIA_QOSTANAY = "Asia/Qostanay"
    ASIA_QYZYLORDA = "Asia/Qyzylorda"
    ASIA_RANGOON = "Asia/Rangoon"
    ASIA_RIYADH = "Asia/Riyadh"
    ASIA_SAKHALIN = "Asia/Sakhalin"
    ASIA_SAMARKAND = "Asia/Samarkand"
    ASIA_SEOUL = "Asia/Seoul"
    ASIA_SHANGHAI = "Asia/Shanghai"
    ASIA_SINGAPORE = "Asia/Singapore"
    ASIA_SREDNEKOLYMSK = "Asia/Srednekolymsk"
    ASIA_TAIPEI = "Asia/Taipei"
    ASIA_TASHKENT = "Asia/Tashkent"
    ASIA_TBILISI = "Asia/Tbilisi"
    ASIA_TEHRAN = "Asia/Tehran"
    ASIA_THIMPHU = "Asia/Thimphu"
    ASIA_TOKYO = "Asia/Tokyo"
    ASIA_TOMSK = "Asia/Tomsk"
    ASIA_ULAANBAATAR = "Asia/Ulaanbaatar"
    ASIA_URUMQI = "Asia/Urumqi"
    ASIA_UST_NERA = "Asia/Ust-Nera"
    ASIA_VIENTIANE = "Asia/Vientiane"
    ASIA_VLADIVOSTOK = "Asia/Vladivostok"
    ASIA_YAKUTSK = "Asia/Yakutsk"
    ASIA_YANGON = "Asia/Yangon"
    ASIA_YEKATERINBURG = "Asia/Yekaterinburg"
    ASIA_YEREVAN = "Asia/Yerevan"
    ATLANTIC_AZORES = "Atlantic/Azores"
    ATLANTIC_BERMUDA = "Atlantic/Bermuda"
    ATLANTIC_CANARY = "Atlantic/Canary"
    ATLANTIC_CAPE_VERDE = "Atlantic/Cape_Verde"
    ATLANTIC_FAROE = "Atlantic/Faroe"
    ATLANTIC_MADEIRA = "Atlantic/Madeira"
    ATLANTIC_REYKJAVIK = "Atlantic/Reykjavik"
    ATLANTIC_SOUTH_GEORGIA = "Atlantic/South_Georgia"
    ATLANTIC_ST_HELENA = "Atlantic/St_Helena"
    ATLANTIC_STANLEY = "Atlantic/Stanley"
    AUSTRALIA_ADELAIDE = "Australia/Adelaide"
    AUSTRALIA_BRISBANE = "Australia/Brisbane"
    AUSTRALIA_BROKEN_HILL = "Australia/Broken_Hill"
    AUSTRALIA_CURRIE = "Australia/Currie"
    AUSTRALIA_DARWIN = "Australia/Darwin"
    AUSTRALIA_EUCLA = "Australia/Eucla"
    AUSTRALIA_HOBART = "Australia/Hobart"
    AUSTRALIA_LINDEMAN = "Australia/Lindeman"
    AUSTRALIA_LORD_HOWE = "Australia/Lord_Howe"
    AUSTRALIA_MELBOURNE = "Australia/Melbourne"
    AUSTRALIA_PERTH = "Australia/Perth"
    AUSTRALIA_SYDNEY = "Australia/Sydney"
    EUROPE_AMSTERDAM = "Europe/Amsterdam"
    EUROPE_ANDORRA = "Europe/Andorra"
    EUROPE_ASTRAKHAN = "Europe/Astrakhan"
    EUROPE_ATHENS = "Europe/Athens"
    EUROPE_BELGRADE = "Europe/Belgrade"
    EUROPE_BERLIN = "Europe/Berlin"
    EUROPE_BRATISLAVA = "Europe/Bratislava"
    EUROPE_BRUSSELS = "Europe/Brussels"
    EUROPE_BUCHAREST = "Europe/Bucharest"
    EUROPE_BUDAPEST = "Europe/Budapest"
    EUROPE_BUSINGEN = "Europe/Busingen"
    EUROPE_CHISINAU = "Europe/Chisinau"
    EUROPE_COPENHAGEN = "Europe/Copenhagen"
    EUROPE_DUBLIN = "Europe/Dublin"
    EUROPE_GIBRALTAR = "Europe/Gibraltar"
    EUROPE_GUERNSEY = "Europe/Guernsey"
    EUROPE_HELSINKI = "Europe/Helsinki"
    EUROPE_ISLE_OF_MAN = "Europe/Isle_of_Man"
    EUROPE_ISTANBUL = "Europe/Istanbul"
    EUROPE_JERSEY = "Europe/Jersey"
    EUROPE_KALININGRAD = "Europe/Kaliningrad"
    EUROPE_KIEV = "Europe/Kiev"
    EUROPE_KIROV = "Europe/Kirov"
    EUROPE_LISBON = "Europe/Lisbon"
    EUROPE_LJUBLJANA = "Europe/Ljubljana"
    EUROPE_LONDON = "Europe/London"
    EUROPE_LUXEMBOURG = "Europe/Luxembourg"
    EUROPE_MADRID = "Europe/Madrid"
    EUROPE_MALTA = "Europe/Malta"
    EUROPE_MARIEHAMN = "Europe/Mariehamn"
    EUROPE_MINSK = "Europe/Minsk"
    EUROPE_MONACO = "Europe/Monaco"
    EUROPE_MOSCOW = "Europe/Moscow"
    EUROPE_NICOSIA = "Europe/Nicosia"
    EUROPE_OSLO = "Europe/Oslo"
    EUROPE_PARIS = "Europe/Paris"
    EUROPE_PODGORICA = "Europe/Podgorica"
    EUROPE_PRAGUE = "Europe/Prague"
    EUROPE_RIGA = "Europe/Riga"
    EUROPE_ROME = "Europe/Rome"
    EUROPE_SAMARA = "Europe/Samara"
    EUROPE_SAN_MARINO = "Europe/San_Marino"
    EUROPE_SARAJEVO = "Europe/Sarajevo"
    EUROPE_SARATOV = "Europe/Saratov"
    EUROPE_SIMFEROPOL = "Europe/Simferopol"
    EUROPE_SKOPJE = "Europe/Skopje"
    EUROPE_SOFIA = "Europe/Sofia"
    EUROPE_STOCKHOLM = "Europe/Stockholm"
    EUROPE_TALLINN = "Europe/Tallinn"
    EUROPE_TIRANE = "Europe/Tirane"
    EUROPE_ULYANOVSK = "Europe/Ulyanovsk"
    EUROPE_UZHGOROD = "Europe/Uzhgorod"
    EUROPE_VADUZ = "Europe/Vaduz"
    EUROPE_VATICAN = "Europe/Vatican"
    EUROPE_VIENNA = "Europe/Vienna"
    EUROPE_VILNIUS = "Europe/Vilnius"
    EUROPE_VOLGOGRAD = "Europe/Volgograd"
    EUROPE_WARSAW = "Europe/Warsaw"
    EUROPE_ZAGREB = "Europe/Zagreb"
    EUROPE_ZAPOROZHYE = "Europe/Zaporozhye"
    EUROPE_ZURICH = "Europe/Zurich"
    INDIAN_ANTANANARIVO = "Indian/Antananarivo"
    INDIAN_CHAGOS = "Indian/Chagos"
    INDIAN_CHRISTMAS = "Indian/Christmas"
    INDIAN_COCOS = "Indian/Cocos"
    INDIAN_COMORO = "Indian/Comoro"
    INDIAN_KERGUELEN = "Indian/Kerguelen"
    INDIAN_MAHE = "Indian/Mahe"
    INDIAN_MALDIVES = "Indian/Maldives"
    INDIAN_MAURITIUS = "Indian/Mauritius"
    INDIAN_MAYOTTE = "Indian/Mayotte"
    INDIAN_REUNION = "Indian/Reunion"
    PACIFIC_APIA = "Pacific/Apia"
    PACIFIC_AUCKLAND = "Pacific/Auckland"
    PACIFIC_BOUGAINVILLE = "Pacific/Bougainville"
    PACIFIC_CHATHAM = "Pacific/Chatham"
    PACIFIC_CHUUK = "Pacific/Chuuk"
    PACIFIC_EASTER = "Pacific/Easter"
    PACIFIC_EFATE = "Pacific/Efate"
    PACIFIC_ENDERBURY = "Pacific/Enderbury"
    PACIFIC_FAKAOFO = "Pacific/Fakaofo"
    PACIFIC_FIJI = "Pacific/Fiji"
    PACIFIC_FUNAFUTI = "Pacific/Funafuti"
    PACIFIC_GALAPAGOS = "Pacific/Galapagos"
    PACIFIC_GAMBIER = "Pacific/Gambier"
    PACIFIC_GUADALCANAL = "Pacific/Guadalcanal"
    PACIFIC_GUAM = "Pacific/Guam"
    PACIFIC_HONOLULU = "Pacific/Honolulu"
    PACIFIC_KANTON = "Pacific/Kanton"
    PACIFIC_KIRITIMATI = "Pacific/Kiritimati"
    PACIFIC_KOSRAE = "Pacific/Kosrae"
    PACIFIC_KWAJALEIN = "Pacific/Kwajalein"
    PACIFIC_MAJURO = "Pacific/Majuro"
    PACIFIC_MARQUESAS = "Pacific/Marquesas"
    PACIFIC_MIDWAY = "Pacific/Midway"
    PACIFIC_NAURU = "Pacific/Nauru"
    PACIFIC_NIUE = "Pacific/Niue"
    PACIFIC_NORFOLK = "Pacific/Norfolk"
    PACIFIC_NOUMEA = "Pacific/Noumea"
    PACIFIC_PAGO_PAGO = "Pacific/Pago_Pago"
    PACIFIC_PALAU = "Pacific/Palau"
    PACIFIC_PITCAIRN = "Pacific/Pitcairn"
    PACIFIC_POHNPEI = "Pacific/Pohnpei"
    PACIFIC_PORT_MORESBY = "Pacific/Port_Moresby"
    PACIFIC_RAROTONGA = "Pacific/Rarotonga"
    PACIFIC_SAIPAN = "Pacific/Saipan"
    PACIFIC_TAHITI = "Pacific/Tahiti"
    PACIFIC_TARAWA = "Pacific/Tarawa"
    PACIFIC_TONGATAPU = "Pacific/Tongatapu"
    PACIFIC_WAKE = "Pacific/Wake"
    PACIFIC_WALLIS = "Pacific/Wallis"


#################################################################
# Schema Management (Tables and Fields)
# NOTE: No delete operations are available in the Airtable API
#################################################################


async def create_table(
    credentials: Credentials,
    base_id: str,
    table_name: str,
    table_fields: list[dict],
) -> dict:
    for field in table_fields:
        assert field.get("name"), "Field name is required"
        assert (
            field.get("type") in TABLE_FIELD_TYPES
        ), f"Field type {field.get('type')} is not valid. Valid types are {TABLE_FIELD_TYPES}."
        # Note fields have differnet options for different types we are not currently validating them

    response = await Requests().post(
        f"https://api.airtable.com/v0/meta/bases/{base_id}/tables",
        headers={"Authorization": credentials.auth_header()},
        json={
            "name": table_name,
            "fields": table_fields,
        },
    )

    return response.json()


async def update_table(
    credentials: Credentials,
    base_id: str,
    table_id: str,
    table_name: str | None = None,
    table_description: str | None = None,
    date_dependency: dict | None = None,
) -> dict:

    assert (
        table_name or table_description or date_dependency
    ), "At least one of table_name, table_description, or date_dependency must be provided"

    params: dict[str, str | dict[str, str]] = {}
    if table_name:
        params["name"] = table_name
    if table_description:
        params["description"] = table_description
    if date_dependency:
        params["dateDependency"] = date_dependency

    response = await Requests().patch(
        f"https://api.airtable.com/v0/meta/bases/{base_id}/tables/{table_id}",
        headers={"Authorization": credentials.auth_header()},
        json=_convert_bools(params),
    )

    return response.json()


async def create_field(
    credentials: Credentials,
    base_id: str,
    table_id: str,
    field_type: TableFieldType,
    name: str,
    description: str | None = None,
    options: dict[str, str] | None = None,
) -> dict[str, str | dict[str, str]]:

    assert (
        field_type in TABLE_FIELD_TYPES
    ), f"Field type {field_type} is not valid. Valid types are {TABLE_FIELD_TYPES}."
    params: dict[str, str | dict[str, str]] = {}
    params["type"] = field_type
    params["name"] = name
    if description:
        params["description"] = description
    if options:
        params["options"] = options

    response = await Requests().post(
        f"https://api.airtable.com/v0/meta/bases/{base_id}/tables/{table_id}/fields",
        headers={"Authorization": credentials.auth_header()},
        json=_convert_bools(params),
    )
    return response.json()


async def update_field(
    credentials: Credentials,
    base_id: str,
    table_id: str,
    field_id: str,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, str]:

    assert name or description, "At least one of name or description must be provided"
    params: dict[str, str | dict[str, str]] = {}
    if name:
        params["name"] = name
    if description:
        params["description"] = description

    response = await Requests().patch(
        f"https://api.airtable.com/v0/meta/bases/{base_id}/tables/{table_id}/fields/{field_id}",
        headers={"Authorization": credentials.auth_header()},
        json=_convert_bools(params),
    )
    return response.json()


#################################################################
# Record Management
#################################################################


async def get_table_schema(
    credentials: Credentials,
    base_id: str,
    table_id_or_name: str,
) -> dict:
    """
    Get the schema for a specific table, including all field definitions.

    Args:
        credentials: Airtable API credentials
        base_id: The base ID
        table_id_or_name: The table ID or name

    Returns:
        Dict containing table schema with fields information
    """
    # First get all tables to find the right one
    response = await Requests().get(
        f"https://api.airtable.com/v0/meta/bases/{base_id}/tables",
        headers={"Authorization": credentials.auth_header()},
    )

    data = response.json()
    tables = data.get("tables", [])

    # Find the matching table
    for table in tables:
        if table.get("id") == table_id_or_name or table.get("name") == table_id_or_name:
            return table

    raise ValueError(f"Table '{table_id_or_name}' not found in base '{base_id}'")


def get_empty_value_for_field(field_type: str) -> Any:
    """
    Return the appropriate empty value for a given Airtable field type.

    Args:
        field_type: The Airtable field type

    Returns:
        The appropriate empty value for that field type
    """
    # Fields that should be false when empty
    if field_type == "checkbox":
        return False

    # Fields that should be empty arrays
    if field_type in [
        "multipleSelects",
        "multipleRecordLinks",
        "multipleAttachments",
        "multipleLookupValues",
        "multipleCollaborators",
    ]:
        return []

    # Fields that should be 0 when empty (numeric types)
    if field_type in [
        "number",
        "percent",
        "currency",
        "rating",
        "duration",
        "count",
        "autoNumber",
    ]:
        return 0

    # Fields that should be empty strings
    if field_type in [
        "singleLineText",
        "multilineText",
        "email",
        "url",
        "phoneNumber",
        "richText",
        "barcode",
    ]:
        return ""

    # Everything else gets null (dates, single selects, formulas, etc.)
    return None


async def normalize_records(
    records: list[dict],
    table_schema: dict,
    include_field_metadata: bool = False,
) -> dict:
    """
    Normalize Airtable records to include all fields with proper empty values.

    Args:
        records: List of record objects from Airtable API
        table_schema: Table schema containing field definitions
        include_field_metadata: Whether to include field metadata in response

    Returns:
        Dict with normalized records and optionally field metadata
    """
    fields = table_schema.get("fields", [])

    # Normalize each record
    normalized_records = []
    for record in records:
        normalized = {
            "id": record.get("id"),
            "createdTime": record.get("createdTime"),
            "fields": {},
        }

        # Add existing fields
        existing_fields = record.get("fields", {})

        # Add all fields from schema, using empty values for missing ones
        for field in fields:
            field_name = field["name"]
            field_type = field["type"]

            if field_name in existing_fields:
                # Field exists, use its value
                normalized["fields"][field_name] = existing_fields[field_name]
            else:
                # Field is missing, add appropriate empty value
                normalized["fields"][field_name] = get_empty_value_for_field(field_type)

        normalized_records.append(normalized)

    # Build result dictionary
    if include_field_metadata:
        field_metadata = {}
        for field in fields:
            metadata = {"type": field["type"], "id": field["id"]}

            # Add type-specific metadata
            options = field.get("options", {})
            if field["type"] == "currency" and "symbol" in options:
                metadata["symbol"] = options["symbol"]
                metadata["precision"] = options.get("precision", 2)
            elif field["type"] == "duration" and "durationFormat" in options:
                metadata["format"] = options["durationFormat"]
            elif field["type"] == "percent" and "precision" in options:
                metadata["precision"] = options["precision"]
            elif (
                field["type"] in ["singleSelect", "multipleSelects"]
                and "choices" in options
            ):
                metadata["choices"] = [choice["name"] for choice in options["choices"]]
            elif field["type"] == "rating" and "max" in options:
                metadata["max"] = options["max"]
                metadata["icon"] = options.get("icon", "star")
                metadata["color"] = options.get("color", "yellowBright")

            field_metadata[field["name"]] = metadata

        return {"records": normalized_records, "field_metadata": field_metadata}
    else:
        return {"records": normalized_records}


async def list_records(
    credentials: Credentials,
    base_id: str,
    table_id_or_name: str,
    # Query parameters
    time_zone: AirtableTimeZones | None = None,
    user_local: str | None = None,
    page_size: int | None = None,
    max_records: int | None = None,
    offset: str | None = None,
    view: str | None = None,
    sort: list[dict[str, str]] | None = None,
    filter_by_formula: str | None = None,
    cell_format: dict[str, str] | None = None,
    fields: list[str] | None = None,
    return_fields_by_field_id: bool | None = None,
    record_metadata: list[str] | None = None,
) -> dict[str, list[dict[str, dict[str, str]]]]:

    params: dict[str, str | dict[str, str] | list[dict[str, str]] | list[str]] = {}
    if time_zone:
        params["timeZone"] = time_zone
    if user_local:
        params["userLocal"] = user_local
    if page_size:
        params["pageSize"] = str(page_size)
    if max_records:
        params["maxRecords"] = str(max_records)
    if offset:
        params["offset"] = offset
    if view:
        params["view"] = view
    if sort:
        params["sort"] = sort
    if filter_by_formula:
        params["filterByFormula"] = filter_by_formula
    if cell_format:
        params["cellFormat"] = cell_format
    if fields:
        params["fields"] = fields
    if return_fields_by_field_id:
        params["returnFieldsByFieldId"] = str(return_fields_by_field_id)
    if record_metadata:
        params["recordMetadata"] = record_metadata

    response = await Requests().get(
        f"https://api.airtable.com/v0/{base_id}/{table_id_or_name}",
        headers={"Authorization": credentials.auth_header()},
        json=_convert_bools(params),
    )
    return response.json()


async def get_record(
    credentials: Credentials,
    base_id: str,
    table_id_or_name: str,
    record_id: str,
) -> dict[str, dict[str, dict[str, str]]]:

    response = await Requests().get(
        f"https://api.airtable.com/v0/{base_id}/{table_id_or_name}/{record_id}",
        headers={"Authorization": credentials.auth_header()},
    )
    return response.json()


async def update_multiple_records(
    credentials: Credentials,
    base_id: str,
    table_id_or_name: str,
    records: list[dict[str, dict[str, str]]],
    perform_upsert: dict[str, list[str]] | None = None,
    return_fields_by_field_id: bool | None = None,
    typecast: bool | None = None,
) -> dict[str, dict[str, dict[str, str]]]:

    params: dict[
        str, str | bool | dict[str, list[str]] | list[dict[str, dict[str, str]]]
    ] = {}
    if perform_upsert:
        params["performUpsert"] = perform_upsert
    if return_fields_by_field_id:
        params["returnFieldsByFieldId"] = str(return_fields_by_field_id)
    if typecast:
        params["typecast"] = typecast

    params["records"] = [_convert_bools(record) for record in records]

    response = await Requests().patch(
        f"https://api.airtable.com/v0/{base_id}/{table_id_or_name}",
        headers={"Authorization": credentials.auth_header()},
        json=_convert_bools(params),
    )
    return response.json()


async def update_record(
    credentials: Credentials,
    base_id: str,
    table_id_or_name: str,
    record_id: str,
    return_fields_by_field_id: bool | None = None,
    typecast: bool | None = None,
    fields: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, str]]]:
    params: dict[str, str | bool | dict[str, Any] | list[dict[str, dict[str, str]]]] = (
        {}
    )
    if return_fields_by_field_id:
        params["returnFieldsByFieldId"] = return_fields_by_field_id
    if typecast:
        params["typecast"] = typecast
    if fields:
        params["fields"] = fields

    response = await Requests().patch(
        f"https://api.airtable.com/v0/{base_id}/{table_id_or_name}/{record_id}",
        headers={"Authorization": credentials.auth_header()},
        json=_convert_bools(params),
    )
    return response.json()


async def create_record(
    credentials: Credentials,
    base_id: str,
    table_id_or_name: str,
    fields: dict[str, Any] | None = None,
    records: list[dict[str, Any]] | None = None,
    return_fields_by_field_id: bool | None = None,
    typecast: bool | None = None,
) -> dict[str, dict[str, dict[str, str]]]:
    assert fields or records, "At least one of fields or records must be provided"
    assert not (fields and records), "Only one of fields or records can be provided"
    if records is not None:
        assert (
            len(records) <= 10
        ), "Only up to 10 records can be provided when using records"

    params: dict[str, str | bool | dict[str, Any] | list[dict[str, Any]]] = {}
    if fields:
        params["fields"] = fields
    if records:
        params["records"] = records
    if return_fields_by_field_id:
        params["returnFieldsByFieldId"] = return_fields_by_field_id
    if typecast:
        params["typecast"] = typecast

    response = await Requests().post(
        f"https://api.airtable.com/v0/{base_id}/{table_id_or_name}",
        headers={"Authorization": credentials.auth_header()},
        json=_convert_bools(params),
    )

    return response.json()


async def delete_multiple_records(
    credentials: Credentials,
    base_id: str,
    table_id_or_name: str,
    records: list[str],
) -> dict[str, dict[str, dict[str, str]]]:

    query_string = "&".join([f"records[]={quote(record)}" for record in records])
    response = await Requests().delete(
        f"https://api.airtable.com/v0/{base_id}/{table_id_or_name}?{query_string}",
        headers={"Authorization": credentials.auth_header()},
    )
    return response.json()


async def delete_record(
    credentials: Credentials,
    base_id: str,
    table_id_or_name: str,
    record_id: str,
) -> dict[str, dict[str, dict[str, str]]]:

    response = await Requests().delete(
        f"https://api.airtable.com/v0/{base_id}/{table_id_or_name}/{record_id}",
        headers={"Authorization": credentials.auth_header()},
    )
    return response.json()


async def create_webhook(
    credentials: Credentials,
    base_id: str,
    webhook_specification: WebhookSpecification,
    notification_url: str | None = None,
) -> Any:

    params: dict[str, Any] = {
        "specification": {
            "options": {
                "filters": webhook_specification.filters.model_dump(exclude_unset=True),
            }
        },
    }
    if webhook_specification.includes:
        params["specification"]["options"]["includes"] = (
            webhook_specification.includes.model_dump(exclude_unset=True)
        )
    if notification_url:
        params["notificationUrl"] = notification_url

    response = await Requests().post(
        f"https://api.airtable.com/v0/bases/{base_id}/webhooks",
        headers={"Authorization": credentials.auth_header()},
        json=_convert_bools(params),
    )
    return response.json()


async def delete_webhook(
    credentials: Credentials,
    base_id: str,
    webhook_id: str,
) -> Any:

    response = await Requests().delete(
        f"https://api.airtable.com/v0/bases/{base_id}/webhooks/{webhook_id}",
        headers={"Authorization": credentials.auth_header()},
    )
    return response.json()


async def list_webhook_payloads(
    credentials: Credentials,
    base_id: str,
    webhook_id: str,
    cursor: str | None = None,
    limit: int | None = None,
) -> ListWebhookPayloadsResponse:

    query_string = ""
    if cursor:
        query_string += f"cursor={cursor}"
    if limit:
        query_string += f"limit={limit}"

    if query_string:
        query_string = f"?{query_string}"

    response = await Requests().get(
        f"https://api.airtable.com/v0/bases/{base_id}/webhooks/{webhook_id}/payloads{query_string}",
        headers={"Authorization": credentials.auth_header()},
    )
    try:
        logger.info(f"Response: {response.json()}")
        return ListWebhookPayloadsResponse(
            payloads=response.json().get("payloads", []),
            cursor=response.json().get("cursor"),
            might_have_more=response.json().get("might_have_more") == "True",
            payloadFormat=response.json().get("payloadFormat", "v0"),
        )
    except Exception as e:
        raise ValueError(
            f"Failed to validate webhook payloads response: {e}\nResponse: {response.json()}"
        )


async def list_webhooks(
    credentials: Credentials,
    base_id: str,
) -> Any:

    response = await Requests().get(
        f"https://api.airtable.com/v0/bases/{base_id}/webhooks",
        headers={"Authorization": credentials.auth_header()},
    )
    return response.json()


class OAuthAuthorizeRequest(BaseModel):
    """OAuth authorization request parameters for Airtable.

    Parameters:
        client_id: An opaque string that identifies your integration with Airtable
        redirect_uri: The URI for the authorize response redirect. Must exactly match a redirect URI
            associated with your integration. HTTPS is required for any URI beside localhost.
        response_type: The string "code"
        scope: A space delimited list of unique scopes. All scopes must be valid Airtable defined scopes
            that have been selected for your integration. At least one scope is required.
        state: A cryptographically generated, opaque string for CSRF protection
        code_challenge: The base64 url-encoding of the sha256 of the code_verifier. Protects against
            man-in-the-middle grant code injection attacks. Part of the PKCE extension of OAuth.
        code_challenge_method: The string "S256"
    """

    client_id: str
    redirect_uri: str
    response_type: str = "code"
    scope: str
    state: str
    code_challenge: str
    code_challenge_method: str = "S256"


class OAuthTokenRequest(BaseModel):
    """OAuth token request parameters for Airtable.

    These parameters must be formatted via application/x-www-form-urlencoded encoding.

    Parameters:
        code: The grant code generated during the authorization request. Can only be used once.
        client_id: The client_id used in the authorization request that generated the code.
            Optional if your integration has a client_secret. Used to prevent MITM attacks.
        redirect_uri: The redirect_uri used in the authorization request that generated the code.
            Used to prevent MITM attacks.
        grant_type: The string "authorization_code".
        code_verifier: A cryptographically generated, opaque string used to generate the
            code_challenge parameter in the authorization request that generated the code.
    """

    code: str
    client_id: str
    redirect_uri: str
    grant_type: str = "authorization_code"
    code_verifier: str


class OAuthRefreshTokenRequest(BaseModel):
    """OAuth token refresh request parameters for Airtable.

    These parameters must be formatted via application/x-www-form-urlencoded encoding.

    Parameters:
        refresh_token: The saved refresh token from the previous token grant.
        client_id: Required if your integration does not have a client_secret.
            Used to prevent MITM attacks.
        grant_type: The string "refresh_token".
        scope: If specified, a subset of the token's existing scopes. Optional.
    """

    refresh_token: str
    client_id: str | None = None
    grant_type: str = "refresh_token"
    scope: str | None = None


class OAuthTokenResponse(BaseModel):
    """OAuth token response from Airtable.

    Successful response has HTTP status code 200 (OK).

    Parameters:
        access_token: An opaque string. Can be used to make requests to the Airtable API on behalf
            of the user, and cannot be recovered if lost.
        refresh_token: An opaque string. Can be used to request a new access token after the current
            one expires.
        token_type: The string "Bearer " (space intentional)
        scope: A string that is a space delimited list of scopes granted to this access token. Can be
            recovered using the get userId and scopes endpoint.
        expires_in: An integer. Time in seconds until the access token expires (expected value is 60 minutes).
        refresh_expires_in: An integer. Time in seconds until the refresh token expires (expected value is 60 days).
    """

    access_token: str
    refresh_token: str
    token_type: str
    scope: str
    expires_in: int
    refresh_expires_in: int


def make_oauth_authorize_url(
    client_id: str,
    redirect_uri: str,
    scopes: list[str],
    state: str,
    code_challenge: str,
) -> str:
    """
    Generate the OAuth authorization URL for Airtable.

    Args:
        client_id: An opaque string that identifies your integration with Airtable
        redirect_uri: The URI for the authorize response redirect
        scope: A space delimited list of unique scopes
        state: A cryptographically generated, opaque string for CSRF protection
        code_challenge: The base64 url-encoding of the sha256 of the code_verifier
        code_challenge_method: The string "S256" (default)
        response_type: The string "code" (default)

    Returns:
        The authorization URL that the user should visit
    """
    # Validate the request parameters
    request_params = OAuthAuthorizeRequest(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=" ".join(scopes),
        state=state,
        code_challenge=code_challenge,
    )

    # Build the authorization URL
    base_url = "https://airtable.com/oauth2/v1/authorize"
    query_string = urlencode(request_params.model_dump(exclude_none=True))

    return f"{base_url}?{query_string}"


async def oauth_exchange_code_for_tokens(
    client_id: str,
    code_verifier: bytes,
    code: str,
    redirect_uri: str,
    client_secret: str | None = None,
) -> OAuthTokenResponse:
    """
    Exchange an authorization code for access and refresh tokens.

    Args:
        client_id: The Airtable integration client ID.
        code_verifier: The original code_verifier (required for PKCE).
        code: The authorization code returned by Airtable.
        redirect_uri: The redirect URI used during authorization.
        client_secret: Integration client secret if available (optional).

    Returns:
        Parsed JSON response containing the access token, refresh token, scope, etc.
    """

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    # Add Authorization header for confidential clients
    if client_secret:
        credentials_encoded = base64.urlsafe_b64encode(
            f"{client_id}:{client_secret}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {credentials_encoded}"

    data = OAuthTokenRequest(
        code=code,
        client_id=client_id,
        redirect_uri=redirect_uri,
        grant_type="authorization_code",
        code_verifier=code_verifier.decode("utf-8"),
    ).model_dump(exclude_none=True)

    response = await Requests().post(
        "https://airtable.com/oauth2/v1/token",
        headers=headers,
        data=data,
    )

    if response.ok:
        return OAuthTokenResponse.model_validate(response.json())
    raise ValueError(
        f"Failed to exchange code for tokens: {response.status} {response.text}"
    )


# NEW helper for refreshing tokens
async def oauth_refresh_tokens(
    client_id: str,
    refresh_token: str,
    client_secret: str | None = None,
) -> OAuthTokenResponse:
    """
    Refresh an expired (or soon-to-expire) access token.

    Args:
        client_id: The Airtable integration client ID.
        refresh_token: The refresh token previously issued by Airtable.
        client_secret: Integration client secret if available (optional).

    Returns:
        Parsed JSON response containing the new tokens and metadata.
        https://airtable.com/oauth2/v1/authorize?client_id=7642abbb-8fbc-494c-b6e0-58484364e28c&redirect_uri=https%3A%2F%2Fdev-builder.agpt.co%2Fauth%2Fintegrations%2Foauth_callback&response_type=code&scope=&state=OcmqX6Y5MTkhHLc6vkbR6uEtSiZHawzEUcxDscqkWRk&code_challenge=v2Ly1CcG8UkCXJ2n--TEKZc6HeKaN1wrZLgIr_qVnJ8&code_challenge_method=S256
    """

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    if client_secret:
        credentials_encoded = base64.urlsafe_b64encode(
            f"{client_id}:{client_secret}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {credentials_encoded}"

    data = OAuthRefreshTokenRequest(
        refresh_token=refresh_token,
        client_id=client_id,
        grant_type="refresh_token",
    ).model_dump(exclude_none=True)

    response = await Requests().post(
        "https://airtable.com/oauth2/v1/token",
        headers=headers,
        data=data,
    )

    if response.ok:
        return OAuthTokenResponse.model_validate(response.json())
    raise ValueError(f"Failed to refresh tokens: {response.status} {response.text}")


#################################################################
# Base Management
#################################################################


async def create_base(
    credentials: Credentials,
    workspace_id: str,
    name: str,
    tables: list[dict] = [
        {
            "description": "Default table",
            "name": "Default table",
            "fields": [
                {
                    "name": "ID",
                    "type": "number",
                    "description": "Auto-incrementing ID field",
                    "options": {"precision": 0},
                }
            ],
        }
    ],
) -> dict:
    """
    Create a new base in Airtable.

    Args:
        credentials: Airtable API credentials
        workspace_id: The workspace ID where the base will be created
        name: The name of the new base
        tables: Optional list of table objects to create in the base

    Returns:
        dict: Response containing the created base information
    """
    params: dict[str, Any] = {
        "name": name,
        "workspaceId": workspace_id,
    }

    if tables:
        params["tables"] = tables

    print(params)

    response = await Requests().post(
        "https://api.airtable.com/v0/meta/bases",
        headers={
            "Authorization": credentials.auth_header(),
            "Content-Type": "application/json",
        },
        json=_convert_bools(params),
    )

    return response.json()


async def list_bases(
    credentials: Credentials,
    offset: str | None = None,
) -> dict:
    """
    List all bases that the authenticated user has access to.

    Args:
        credentials: Airtable API credentials
        offset: Optional pagination offset

    Returns:
        dict: Response containing the list of bases
    """
    params = {}
    if offset:
        params["offset"] = offset

    response = await Requests().get(
        "https://api.airtable.com/v0/meta/bases",
        headers={"Authorization": credentials.auth_header()},
        params=params,
    )

    return response.json()


async def get_base_tables(
    credentials: Credentials,
    base_id: str,
) -> list[dict]:
    """
    Get all tables for a specific base.

    Args:
        credentials: Airtable API credentials
        base_id: The ID of the base

    Returns:
        list[dict]: List of table objects with their schemas
    """
    response = await Requests().get(
        f"https://api.airtable.com/v0/meta/bases/{base_id}/tables",
        headers={"Authorization": credentials.auth_header()},
    )

    data = response.json()
    return data.get("tables", [])
