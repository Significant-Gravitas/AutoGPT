/*
Breaks a Javascript string into individual user-perceived "characters" 
called extended grapheme clusters by implementing the Unicode UAX-29 standard, version 10.0.0

Usage:
var splitter = new GraphemeSplitter();
//returns an array of strings, one string for each grapheme cluster
var graphemes = splitter.splitGraphemes(string); 

*/
function GraphemeSplitter(){
	var CR = 0,
		LF = 1,
		Control = 2,
		Extend = 3,
		Regional_Indicator = 4,
		SpacingMark = 5,
		L = 6,
		V = 7,
		T = 8,
		LV = 9,
		LVT = 10,
		Other = 11,
		Prepend = 12,
		E_Base = 13,
		E_Modifier = 14,
		ZWJ = 15,
		Glue_After_Zwj = 16,
		E_Base_GAZ = 17;
		
	// BreakTypes
	var NotBreak = 0,
		BreakStart = 1,
		Break = 2,
		BreakLastRegional = 3,
		BreakPenultimateRegional = 4;
		
	function isSurrogate(str, pos) {
		return  0xd800 <= str.charCodeAt(pos) && str.charCodeAt(pos) <= 0xdbff && 
				0xdc00 <= str.charCodeAt(pos + 1) && str.charCodeAt(pos + 1) <= 0xdfff;
	}
		
	// Private function, gets a Unicode code point from a JavaScript UTF-16 string
	// handling surrogate pairs appropriately
	function codePointAt(str, idx){
		if(idx === undefined){
			idx = 0;
		}
		var code = str.charCodeAt(idx);

		// if a high surrogate
		if (0xD800 <= code && code <= 0xDBFF && 
			idx < str.length - 1){
			var hi = code;
			var low = str.charCodeAt(idx + 1);
			if (0xDC00 <= low && low <= 0xDFFF){
				return ((hi - 0xD800) * 0x400) + (low - 0xDC00) + 0x10000;
			}
			return hi;
		}
		
		// if a low surrogate
		if (0xDC00 <= code && code <= 0xDFFF &&
			idx >= 1){
			var hi = str.charCodeAt(idx - 1);
			var low = code;
			if (0xD800 <= hi && hi <= 0xDBFF){
				return ((hi - 0xD800) * 0x400) + (low - 0xDC00) + 0x10000;
			}
			return low;
		}
		
		//just return the char if an unmatched surrogate half or a 
		//single-char codepoint
		return code;
	}
	
	// Private function, returns whether a break is allowed between the 
	// two given grapheme breaking classes
	function shouldBreak(start, mid, end){
		var all = [start].concat(mid).concat([end]);
		var previous = all[all.length - 2]
		var next = end
		
		// Lookahead termintor for:
		// GB10. (E_Base | EBG) Extend* ?	E_Modifier
		var eModifierIndex = all.lastIndexOf(E_Modifier)
		if(eModifierIndex > 1 &&
			all.slice(1, eModifierIndex).every(function(c){return c == Extend}) &&
			[Extend, E_Base, E_Base_GAZ].indexOf(start) == -1){
			return Break
		}

		// Lookahead termintor for:
		// GB12. ^ (RI RI)* RI	?	RI
		// GB13. [^RI] (RI RI)* RI	?	RI
		var rIIndex = all.lastIndexOf(Regional_Indicator)
		if(rIIndex > 0 &&
			all.slice(1, rIIndex).every(function(c){return c == Regional_Indicator}) &&
			[Prepend, Regional_Indicator].indexOf(previous) == -1) { 
			if(all.filter(function(c){return c == Regional_Indicator}).length % 2 == 1) {
				return BreakLastRegional
			}
			else {
				return BreakPenultimateRegional
			}
		}
		
		// GB3. CR X LF
		if(previous == CR && next == LF){
			return NotBreak;
		}
		// GB4. (Control|CR|LF) ÷
		else if(previous == Control || previous == CR || previous == LF){
			if(next == E_Modifier && mid.every(function(c){return c == Extend})){
				return Break
			}
			else {
				return BreakStart
			}
		}
		// GB5. ÷ (Control|CR|LF)
		else if(next == Control || next == CR || next == LF){
			return BreakStart;
		}
		// GB6. L X (L|V|LV|LVT)
		else if(previous == L && 
			(next == L || next == V || next == LV || next == LVT)){
			return NotBreak;
		}
		// GB7. (LV|V) X (V|T)
		else if((previous == LV || previous == V) && 
			(next == V || next == T)){
			return NotBreak;
		}
		// GB8. (LVT|T) X (T)
		else if((previous == LVT || previous == T) && 
			next == T){
			return NotBreak;
		}
		// GB9. X (Extend|ZWJ)
		else if (next == Extend || next == ZWJ){
			return NotBreak;
		}
		// GB9a. X SpacingMark
		else if(next == SpacingMark){
			return NotBreak;
		}
		// GB9b. Prepend X
		else if (previous == Prepend){
			return NotBreak;
		}
		
		// GB10. (E_Base | EBG) Extend* ?	E_Modifier
		var previousNonExtendIndex = all.indexOf(Extend) != -1 ? all.lastIndexOf(Extend) - 1 : all.length - 2;
		if([E_Base, E_Base_GAZ].indexOf(all[previousNonExtendIndex]) != -1 &&
			all.slice(previousNonExtendIndex + 1, -1).every(function(c){return c == Extend}) &&
			next == E_Modifier){
			return NotBreak;
		}
		
		// GB11. ZWJ ? (Glue_After_Zwj | EBG)
		if(previous == ZWJ && [Glue_After_Zwj, E_Base_GAZ].indexOf(next) != -1) {
			return NotBreak;
		}

		// GB12. ^ (RI RI)* RI ? RI
		// GB13. [^RI] (RI RI)* RI ? RI
		if(mid.indexOf(Regional_Indicator) != -1) { 
			return Break;
		}
		if(previous == Regional_Indicator && next == Regional_Indicator) {
			return NotBreak;
		}

		// GB999. Any ? Any
		return BreakStart;
	}
	
	// Returns the next grapheme break in the string after the given index
	this.nextBreak = function(string, index){
		if(index === undefined){
			index = 0;
		}
		if(index < 0){
			return 0;
		}
		if(index >= string.length - 1){
			return string.length;
		}
		var prev = getGraphemeBreakProperty(codePointAt(string, index));
		var mid = []
		for (var i = index + 1; i < string.length; i++) {
			// check for already processed low surrogates
			if(isSurrogate(string, i - 1)){
				continue;
			}
		
			var next = getGraphemeBreakProperty(codePointAt(string, i));
			if(shouldBreak(prev, mid, next)){
				return i;
			}
			
			mid.push(next);
		}
		return string.length;
	};
	
	// Breaks the given string into an array of grapheme cluster strings
	this.splitGraphemes = function(str){
		var res = [];
		var index = 0;
		var brk;
		while((brk = this.nextBreak(str, index)) < str.length){
			res.push(str.slice(index, brk));
			index = brk;
		}
		if(index < str.length){
			res.push(str.slice(index));
		}
		return res;
	};

	// Returns the iterator of grapheme clusters there are in the given string
	this.iterateGraphemes = function(str) {
		var index = 0;
		var res = {
			next: (function() {
				var value;
				var brk;
				if ((brk = this.nextBreak(str, index)) < str.length) {
					value = str.slice(index, brk);
					index = brk;
					return { value: value, done: false };
				}
				if (index < str.length) {
					value = str.slice(index);
					index = str.length;
					return { value: value, done: false };
				}
				return { value: undefined, done: true };
			}).bind(this)
		};
		// ES2015 @@iterator method (iterable) for spread syntax and for...of statement
		if (typeof Symbol !== 'undefined' && Symbol.iterator) {
			res[Symbol.iterator] = function() {return res};
		}
		return res;
	};

	// Returns the number of grapheme clusters there are in the given string
	this.countGraphemes = function(str){
		var count = 0;
		var index = 0;
		var brk;
		while((brk = this.nextBreak(str, index)) < str.length){
			index = brk;
			count++;
		}
		if(index < str.length){
			count++;
		}
		return count;
	};
	
	//given a Unicode code point, determines this symbol's grapheme break property
	function getGraphemeBreakProperty(code){
		
		//grapheme break property for Unicode 10.0.0, 
		//taken from http://www.unicode.org/Public/10.0.0/ucd/auxiliary/GraphemeBreakProperty.txt
		//and adapted to JavaScript rules
		
		if(		
		(0x0600 <= code && code <= 0x0605) || // Cf   [6] ARABIC NUMBER SIGN..ARABIC NUMBER MARK ABOVE
		0x06DD == code || // Cf       ARABIC END OF AYAH
		0x070F == code || // Cf       SYRIAC ABBREVIATION MARK
		0x08E2 == code || // Cf       ARABIC DISPUTED END OF AYAH
		0x0D4E == code || // Lo       MALAYALAM LETTER DOT REPH
		0x110BD == code || // Cf       KAITHI NUMBER SIGN
		(0x111C2 <= code && code <= 0x111C3) || // Lo   [2] SHARADA SIGN JIHVAMULIYA..SHARADA SIGN UPADHMANIYA
		0x11A3A == code || // Lo       ZANABAZAR SQUARE CLUSTER-INITIAL LETTER RA
		(0x11A86 <= code && code <= 0x11A89) || // Lo   [4] SOYOMBO CLUSTER-INITIAL LETTER RA..SOYOMBO CLUSTER-INITIAL LETTER SA
		0x11D46 == code // Lo       MASARAM GONDI REPHA
		){
			return Prepend;
		}
		if(
		0x000D == code // Cc       <control-000D>
		){
			return CR;
		}
		
		if(
		0x000A == code // Cc       <control-000A>
		){
			return LF;
		}
		
		
		if(
		(0x0000 <= code && code <= 0x0009) || // Cc  [10] <control-0000>..<control-0009>
		(0x000B <= code && code <= 0x000C) || // Cc   [2] <control-000B>..<control-000C>
		(0x000E <= code && code <= 0x001F) || // Cc  [18] <control-000E>..<control-001F>
		(0x007F <= code && code <= 0x009F) || // Cc  [33] <control-007F>..<control-009F>
		0x00AD == code || // Cf       SOFT HYPHEN
		0x061C == code || // Cf       ARABIC LETTER MARK
	
		0x180E == code || // Cf       MONGOLIAN VOWEL SEPARATOR
		0x200B == code || // Cf       ZERO WIDTH SPACE
		(0x200E <= code && code <= 0x200F) || // Cf   [2] LEFT-TO-RIGHT MARK..RIGHT-TO-LEFT MARK
		0x2028 == code || // Zl       LINE SEPARATOR
		0x2029 == code || // Zp       PARAGRAPH SEPARATOR
		(0x202A <= code && code <= 0x202E) || // Cf   [5] LEFT-TO-RIGHT EMBEDDING..RIGHT-TO-LEFT OVERRIDE
		(0x2060 <= code && code <= 0x2064) || // Cf   [5] WORD JOINER..INVISIBLE PLUS
		0x2065 == code || // Cn       <reserved-2065>
		(0x2066 <= code && code <= 0x206F) || // Cf  [10] LEFT-TO-RIGHT ISOLATE..NOMINAL DIGIT SHAPES
		(0xD800 <= code && code <= 0xDFFF) || // Cs [2048] <surrogate-D800>..<surrogate-DFFF>
		0xFEFF == code || // Cf       ZERO WIDTH NO-BREAK SPACE
		(0xFFF0 <= code && code <= 0xFFF8) || // Cn   [9] <reserved-FFF0>..<reserved-FFF8>
		(0xFFF9 <= code && code <= 0xFFFB) || // Cf   [3] INTERLINEAR ANNOTATION ANCHOR..INTERLINEAR ANNOTATION TERMINATOR
		(0x1BCA0 <= code && code <= 0x1BCA3) || // Cf   [4] SHORTHAND FORMAT LETTER OVERLAP..SHORTHAND FORMAT UP STEP
		(0x1D173 <= code && code <= 0x1D17A) || // Cf   [8] MUSICAL SYMBOL BEGIN BEAM..MUSICAL SYMBOL END PHRASE
		0xE0000 == code || // Cn       <reserved-E0000>
		0xE0001 == code || // Cf       LANGUAGE TAG
		(0xE0002 <= code && code <= 0xE001F) || // Cn  [30] <reserved-E0002>..<reserved-E001F>
		(0xE0080 <= code && code <= 0xE00FF) || // Cn [128] <reserved-E0080>..<reserved-E00FF>
		(0xE01F0 <= code && code <= 0xE0FFF) // Cn [3600] <reserved-E01F0>..<reserved-E0FFF>
		){
			return Control;
		}
		
		
		if(
		(0x0300 <= code && code <= 0x036F) || // Mn [112] COMBINING GRAVE ACCENT..COMBINING LATIN SMALL LETTER X
		(0x0483 <= code && code <= 0x0487) || // Mn   [5] COMBINING CYRILLIC TITLO..COMBINING CYRILLIC POKRYTIE
		(0x0488 <= code && code <= 0x0489) || // Me   [2] COMBINING CYRILLIC HUNDRED THOUSANDS SIGN..COMBINING CYRILLIC MILLIONS SIGN
		(0x0591 <= code && code <= 0x05BD) || // Mn  [45] HEBREW ACCENT ETNAHTA..HEBREW POINT METEG
		0x05BF == code || // Mn       HEBREW POINT RAFE
		(0x05C1 <= code && code <= 0x05C2) || // Mn   [2] HEBREW POINT SHIN DOT..HEBREW POINT SIN DOT
		(0x05C4 <= code && code <= 0x05C5) || // Mn   [2] HEBREW MARK UPPER DOT..HEBREW MARK LOWER DOT
		0x05C7 == code || // Mn       HEBREW POINT QAMATS QATAN
		(0x0610 <= code && code <= 0x061A) || // Mn  [11] ARABIC SIGN SALLALLAHOU ALAYHE WASSALLAM..ARABIC SMALL KASRA
		(0x064B <= code && code <= 0x065F) || // Mn  [21] ARABIC FATHATAN..ARABIC WAVY HAMZA BELOW
		0x0670 == code || // Mn       ARABIC LETTER SUPERSCRIPT ALEF
		(0x06D6 <= code && code <= 0x06DC) || // Mn   [7] ARABIC SMALL HIGH LIGATURE SAD WITH LAM WITH ALEF MAKSURA..ARABIC SMALL HIGH SEEN
		(0x06DF <= code && code <= 0x06E4) || // Mn   [6] ARABIC SMALL HIGH ROUNDED ZERO..ARABIC SMALL HIGH MADDA
		(0x06E7 <= code && code <= 0x06E8) || // Mn   [2] ARABIC SMALL HIGH YEH..ARABIC SMALL HIGH NOON
		(0x06EA <= code && code <= 0x06ED) || // Mn   [4] ARABIC EMPTY CENTRE LOW STOP..ARABIC SMALL LOW MEEM
		0x0711 == code || // Mn       SYRIAC LETTER SUPERSCRIPT ALAPH
		(0x0730 <= code && code <= 0x074A) || // Mn  [27] SYRIAC PTHAHA ABOVE..SYRIAC BARREKH
		(0x07A6 <= code && code <= 0x07B0) || // Mn  [11] THAANA ABAFILI..THAANA SUKUN
		(0x07EB <= code && code <= 0x07F3) || // Mn   [9] NKO COMBINING SHORT HIGH TONE..NKO COMBINING DOUBLE DOT ABOVE
		(0x0816 <= code && code <= 0x0819) || // Mn   [4] SAMARITAN MARK IN..SAMARITAN MARK DAGESH
		(0x081B <= code && code <= 0x0823) || // Mn   [9] SAMARITAN MARK EPENTHETIC YUT..SAMARITAN VOWEL SIGN A
		(0x0825 <= code && code <= 0x0827) || // Mn   [3] SAMARITAN VOWEL SIGN SHORT A..SAMARITAN VOWEL SIGN U
		(0x0829 <= code && code <= 0x082D) || // Mn   [5] SAMARITAN VOWEL SIGN LONG I..SAMARITAN MARK NEQUDAA
		(0x0859 <= code && code <= 0x085B) || // Mn   [3] MANDAIC AFFRICATION MARK..MANDAIC GEMINATION MARK
		(0x08D4 <= code && code <= 0x08E1) || // Mn  [14] ARABIC SMALL HIGH WORD AR-RUB..ARABIC SMALL HIGH SIGN SAFHA
		(0x08E3 <= code && code <= 0x0902) || // Mn  [32] ARABIC TURNED DAMMA BELOW..DEVANAGARI SIGN ANUSVARA
		0x093A == code || // Mn       DEVANAGARI VOWEL SIGN OE
		0x093C == code || // Mn       DEVANAGARI SIGN NUKTA
		(0x0941 <= code && code <= 0x0948) || // Mn   [8] DEVANAGARI VOWEL SIGN U..DEVANAGARI VOWEL SIGN AI
		0x094D == code || // Mn       DEVANAGARI SIGN VIRAMA
		(0x0951 <= code && code <= 0x0957) || // Mn   [7] DEVANAGARI STRESS SIGN UDATTA..DEVANAGARI VOWEL SIGN UUE
		(0x0962 <= code && code <= 0x0963) || // Mn   [2] DEVANAGARI VOWEL SIGN VOCALIC L..DEVANAGARI VOWEL SIGN VOCALIC LL
		0x0981 == code || // Mn       BENGALI SIGN CANDRABINDU
		0x09BC == code || // Mn       BENGALI SIGN NUKTA
		0x09BE == code || // Mc       BENGALI VOWEL SIGN AA
		(0x09C1 <= code && code <= 0x09C4) || // Mn   [4] BENGALI VOWEL SIGN U..BENGALI VOWEL SIGN VOCALIC RR
		0x09CD == code || // Mn       BENGALI SIGN VIRAMA
		0x09D7 == code || // Mc       BENGALI AU LENGTH MARK
		(0x09E2 <= code && code <= 0x09E3) || // Mn   [2] BENGALI VOWEL SIGN VOCALIC L..BENGALI VOWEL SIGN VOCALIC LL
		(0x0A01 <= code && code <= 0x0A02) || // Mn   [2] GURMUKHI SIGN ADAK BINDI..GURMUKHI SIGN BINDI
		0x0A3C == code || // Mn       GURMUKHI SIGN NUKTA
		(0x0A41 <= code && code <= 0x0A42) || // Mn   [2] GURMUKHI VOWEL SIGN U..GURMUKHI VOWEL SIGN UU
		(0x0A47 <= code && code <= 0x0A48) || // Mn   [2] GURMUKHI VOWEL SIGN EE..GURMUKHI VOWEL SIGN AI
		(0x0A4B <= code && code <= 0x0A4D) || // Mn   [3] GURMUKHI VOWEL SIGN OO..GURMUKHI SIGN VIRAMA
		0x0A51 == code || // Mn       GURMUKHI SIGN UDAAT
		(0x0A70 <= code && code <= 0x0A71) || // Mn   [2] GURMUKHI TIPPI..GURMUKHI ADDAK
		0x0A75 == code || // Mn       GURMUKHI SIGN YAKASH
		(0x0A81 <= code && code <= 0x0A82) || // Mn   [2] GUJARATI SIGN CANDRABINDU..GUJARATI SIGN ANUSVARA
		0x0ABC == code || // Mn       GUJARATI SIGN NUKTA
		(0x0AC1 <= code && code <= 0x0AC5) || // Mn   [5] GUJARATI VOWEL SIGN U..GUJARATI VOWEL SIGN CANDRA E
		(0x0AC7 <= code && code <= 0x0AC8) || // Mn   [2] GUJARATI VOWEL SIGN E..GUJARATI VOWEL SIGN AI
		0x0ACD == code || // Mn       GUJARATI SIGN VIRAMA
		(0x0AE2 <= code && code <= 0x0AE3) || // Mn   [2] GUJARATI VOWEL SIGN VOCALIC L..GUJARATI VOWEL SIGN VOCALIC LL
		(0x0AFA <= code && code <= 0x0AFF) || // Mn   [6] GUJARATI SIGN SUKUN..GUJARATI SIGN TWO-CIRCLE NUKTA ABOVE
		0x0B01 == code || // Mn       ORIYA SIGN CANDRABINDU
		0x0B3C == code || // Mn       ORIYA SIGN NUKTA
		0x0B3E == code || // Mc       ORIYA VOWEL SIGN AA
		0x0B3F == code || // Mn       ORIYA VOWEL SIGN I
		(0x0B41 <= code && code <= 0x0B44) || // Mn   [4] ORIYA VOWEL SIGN U..ORIYA VOWEL SIGN VOCALIC RR
		0x0B4D == code || // Mn       ORIYA SIGN VIRAMA
		0x0B56 == code || // Mn       ORIYA AI LENGTH MARK
		0x0B57 == code || // Mc       ORIYA AU LENGTH MARK
		(0x0B62 <= code && code <= 0x0B63) || // Mn   [2] ORIYA VOWEL SIGN VOCALIC L..ORIYA VOWEL SIGN VOCALIC LL
		0x0B82 == code || // Mn       TAMIL SIGN ANUSVARA
		0x0BBE == code || // Mc       TAMIL VOWEL SIGN AA
		0x0BC0 == code || // Mn       TAMIL VOWEL SIGN II
		0x0BCD == code || // Mn       TAMIL SIGN VIRAMA
		0x0BD7 == code || // Mc       TAMIL AU LENGTH MARK
		0x0C00 == code || // Mn       TELUGU SIGN COMBINING CANDRABINDU ABOVE
		(0x0C3E <= code && code <= 0x0C40) || // Mn   [3] TELUGU VOWEL SIGN AA..TELUGU VOWEL SIGN II
		(0x0C46 <= code && code <= 0x0C48) || // Mn   [3] TELUGU VOWEL SIGN E..TELUGU VOWEL SIGN AI
		(0x0C4A <= code && code <= 0x0C4D) || // Mn   [4] TELUGU VOWEL SIGN O..TELUGU SIGN VIRAMA
		(0x0C55 <= code && code <= 0x0C56) || // Mn   [2] TELUGU LENGTH MARK..TELUGU AI LENGTH MARK
		(0x0C62 <= code && code <= 0x0C63) || // Mn   [2] TELUGU VOWEL SIGN VOCALIC L..TELUGU VOWEL SIGN VOCALIC LL
		0x0C81 == code || // Mn       KANNADA SIGN CANDRABINDU
		0x0CBC == code || // Mn       KANNADA SIGN NUKTA
		0x0CBF == code || // Mn       KANNADA VOWEL SIGN I
		0x0CC2 == code || // Mc       KANNADA VOWEL SIGN UU
		0x0CC6 == code || // Mn       KANNADA VOWEL SIGN E
		(0x0CCC <= code && code <= 0x0CCD) || // Mn   [2] KANNADA VOWEL SIGN AU..KANNADA SIGN VIRAMA
		(0x0CD5 <= code && code <= 0x0CD6) || // Mc   [2] KANNADA LENGTH MARK..KANNADA AI LENGTH MARK
		(0x0CE2 <= code && code <= 0x0CE3) || // Mn   [2] KANNADA VOWEL SIGN VOCALIC L..KANNADA VOWEL SIGN VOCALIC LL
		(0x0D00 <= code && code <= 0x0D01) || // Mn   [2] MALAYALAM SIGN COMBINING ANUSVARA ABOVE..MALAYALAM SIGN CANDRABINDU
		(0x0D3B <= code && code <= 0x0D3C) || // Mn   [2] MALAYALAM SIGN VERTICAL BAR VIRAMA..MALAYALAM SIGN CIRCULAR VIRAMA
		0x0D3E == code || // Mc       MALAYALAM VOWEL SIGN AA
		(0x0D41 <= code && code <= 0x0D44) || // Mn   [4] MALAYALAM VOWEL SIGN U..MALAYALAM VOWEL SIGN VOCALIC RR
		0x0D4D == code || // Mn       MALAYALAM SIGN VIRAMA
		0x0D57 == code || // Mc       MALAYALAM AU LENGTH MARK
		(0x0D62 <= code && code <= 0x0D63) || // Mn   [2] MALAYALAM VOWEL SIGN VOCALIC L..MALAYALAM VOWEL SIGN VOCALIC LL
		0x0DCA == code || // Mn       SINHALA SIGN AL-LAKUNA
		0x0DCF == code || // Mc       SINHALA VOWEL SIGN AELA-PILLA
		(0x0DD2 <= code && code <= 0x0DD4) || // Mn   [3] SINHALA VOWEL SIGN KETTI IS-PILLA..SINHALA VOWEL SIGN KETTI PAA-PILLA
		0x0DD6 == code || // Mn       SINHALA VOWEL SIGN DIGA PAA-PILLA
		0x0DDF == code || // Mc       SINHALA VOWEL SIGN GAYANUKITTA
		0x0E31 == code || // Mn       THAI CHARACTER MAI HAN-AKAT
		(0x0E34 <= code && code <= 0x0E3A) || // Mn   [7] THAI CHARACTER SARA I..THAI CHARACTER PHINTHU
		(0x0E47 <= code && code <= 0x0E4E) || // Mn   [8] THAI CHARACTER MAITAIKHU..THAI CHARACTER YAMAKKAN
		0x0EB1 == code || // Mn       LAO VOWEL SIGN MAI KAN
		(0x0EB4 <= code && code <= 0x0EB9) || // Mn   [6] LAO VOWEL SIGN I..LAO VOWEL SIGN UU
		(0x0EBB <= code && code <= 0x0EBC) || // Mn   [2] LAO VOWEL SIGN MAI KON..LAO SEMIVOWEL SIGN LO
		(0x0EC8 <= code && code <= 0x0ECD) || // Mn   [6] LAO TONE MAI EK..LAO NIGGAHITA
		(0x0F18 <= code && code <= 0x0F19) || // Mn   [2] TIBETAN ASTROLOGICAL SIGN -KHYUD PA..TIBETAN ASTROLOGICAL SIGN SDONG TSHUGS
		0x0F35 == code || // Mn       TIBETAN MARK NGAS BZUNG NYI ZLA
		0x0F37 == code || // Mn       TIBETAN MARK NGAS BZUNG SGOR RTAGS
		0x0F39 == code || // Mn       TIBETAN MARK TSA -PHRU
		(0x0F71 <= code && code <= 0x0F7E) || // Mn  [14] TIBETAN VOWEL SIGN AA..TIBETAN SIGN RJES SU NGA RO
		(0x0F80 <= code && code <= 0x0F84) || // Mn   [5] TIBETAN VOWEL SIGN REVERSED I..TIBETAN MARK HALANTA
		(0x0F86 <= code && code <= 0x0F87) || // Mn   [2] TIBETAN SIGN LCI RTAGS..TIBETAN SIGN YANG RTAGS
		(0x0F8D <= code && code <= 0x0F97) || // Mn  [11] TIBETAN SUBJOINED SIGN LCE TSA CAN..TIBETAN SUBJOINED LETTER JA
		(0x0F99 <= code && code <= 0x0FBC) || // Mn  [36] TIBETAN SUBJOINED LETTER NYA..TIBETAN SUBJOINED LETTER FIXED-FORM RA
		0x0FC6 == code || // Mn       TIBETAN SYMBOL PADMA GDAN
		(0x102D <= code && code <= 0x1030) || // Mn   [4] MYANMAR VOWEL SIGN I..MYANMAR VOWEL SIGN UU
		(0x1032 <= code && code <= 0x1037) || // Mn   [6] MYANMAR VOWEL SIGN AI..MYANMAR SIGN DOT BELOW
		(0x1039 <= code && code <= 0x103A) || // Mn   [2] MYANMAR SIGN VIRAMA..MYANMAR SIGN ASAT
		(0x103D <= code && code <= 0x103E) || // Mn   [2] MYANMAR CONSONANT SIGN MEDIAL WA..MYANMAR CONSONANT SIGN MEDIAL HA
		(0x1058 <= code && code <= 0x1059) || // Mn   [2] MYANMAR VOWEL SIGN VOCALIC L..MYANMAR VOWEL SIGN VOCALIC LL
		(0x105E <= code && code <= 0x1060) || // Mn   [3] MYANMAR CONSONANT SIGN MON MEDIAL NA..MYANMAR CONSONANT SIGN MON MEDIAL LA
		(0x1071 <= code && code <= 0x1074) || // Mn   [4] MYANMAR VOWEL SIGN GEBA KAREN I..MYANMAR VOWEL SIGN KAYAH EE
		0x1082 == code || // Mn       MYANMAR CONSONANT SIGN SHAN MEDIAL WA
		(0x1085 <= code && code <= 0x1086) || // Mn   [2] MYANMAR VOWEL SIGN SHAN E ABOVE..MYANMAR VOWEL SIGN SHAN FINAL Y
		0x108D == code || // Mn       MYANMAR SIGN SHAN COUNCIL EMPHATIC TONE
		0x109D == code || // Mn       MYANMAR VOWEL SIGN AITON AI
		(0x135D <= code && code <= 0x135F) || // Mn   [3] ETHIOPIC COMBINING GEMINATION AND VOWEL LENGTH MARK..ETHIOPIC COMBINING GEMINATION MARK
		(0x1712 <= code && code <= 0x1714) || // Mn   [3] TAGALOG VOWEL SIGN I..TAGALOG SIGN VIRAMA
		(0x1732 <= code && code <= 0x1734) || // Mn   [3] HANUNOO VOWEL SIGN I..HANUNOO SIGN PAMUDPOD
		(0x1752 <= code && code <= 0x1753) || // Mn   [2] BUHID VOWEL SIGN I..BUHID VOWEL SIGN U
		(0x1772 <= code && code <= 0x1773) || // Mn   [2] TAGBANWA VOWEL SIGN I..TAGBANWA VOWEL SIGN U
		(0x17B4 <= code && code <= 0x17B5) || // Mn   [2] KHMER VOWEL INHERENT AQ..KHMER VOWEL INHERENT AA
		(0x17B7 <= code && code <= 0x17BD) || // Mn   [7] KHMER VOWEL SIGN I..KHMER VOWEL SIGN UA
		0x17C6 == code || // Mn       KHMER SIGN NIKAHIT
		(0x17C9 <= code && code <= 0x17D3) || // Mn  [11] KHMER SIGN MUUSIKATOAN..KHMER SIGN BATHAMASAT
		0x17DD == code || // Mn       KHMER SIGN ATTHACAN
		(0x180B <= code && code <= 0x180D) || // Mn   [3] MONGOLIAN FREE VARIATION SELECTOR ONE..MONGOLIAN FREE VARIATION SELECTOR THREE
		(0x1885 <= code && code <= 0x1886) || // Mn   [2] MONGOLIAN LETTER ALI GALI BALUDA..MONGOLIAN LETTER ALI GALI THREE BALUDA
		0x18A9 == code || // Mn       MONGOLIAN LETTER ALI GALI DAGALGA
		(0x1920 <= code && code <= 0x1922) || // Mn   [3] LIMBU VOWEL SIGN A..LIMBU VOWEL SIGN U
		(0x1927 <= code && code <= 0x1928) || // Mn   [2] LIMBU VOWEL SIGN E..LIMBU VOWEL SIGN O
		0x1932 == code || // Mn       LIMBU SMALL LETTER ANUSVARA
		(0x1939 <= code && code <= 0x193B) || // Mn   [3] LIMBU SIGN MUKPHRENG..LIMBU SIGN SA-I
		(0x1A17 <= code && code <= 0x1A18) || // Mn   [2] BUGINESE VOWEL SIGN I..BUGINESE VOWEL SIGN U
		0x1A1B == code || // Mn       BUGINESE VOWEL SIGN AE
		0x1A56 == code || // Mn       TAI THAM CONSONANT SIGN MEDIAL LA
		(0x1A58 <= code && code <= 0x1A5E) || // Mn   [7] TAI THAM SIGN MAI KANG LAI..TAI THAM CONSONANT SIGN SA
		0x1A60 == code || // Mn       TAI THAM SIGN SAKOT
		0x1A62 == code || // Mn       TAI THAM VOWEL SIGN MAI SAT
		(0x1A65 <= code && code <= 0x1A6C) || // Mn   [8] TAI THAM VOWEL SIGN I..TAI THAM VOWEL SIGN OA BELOW
		(0x1A73 <= code && code <= 0x1A7C) || // Mn  [10] TAI THAM VOWEL SIGN OA ABOVE..TAI THAM SIGN KHUEN-LUE KARAN
		0x1A7F == code || // Mn       TAI THAM COMBINING CRYPTOGRAMMIC DOT
		(0x1AB0 <= code && code <= 0x1ABD) || // Mn  [14] COMBINING DOUBLED CIRCUMFLEX ACCENT..COMBINING PARENTHESES BELOW
		0x1ABE == code || // Me       COMBINING PARENTHESES OVERLAY
		(0x1B00 <= code && code <= 0x1B03) || // Mn   [4] BALINESE SIGN ULU RICEM..BALINESE SIGN SURANG
		0x1B34 == code || // Mn       BALINESE SIGN REREKAN
		(0x1B36 <= code && code <= 0x1B3A) || // Mn   [5] BALINESE VOWEL SIGN ULU..BALINESE VOWEL SIGN RA REPA
		0x1B3C == code || // Mn       BALINESE VOWEL SIGN LA LENGA
		0x1B42 == code || // Mn       BALINESE VOWEL SIGN PEPET
		(0x1B6B <= code && code <= 0x1B73) || // Mn   [9] BALINESE MUSICAL SYMBOL COMBINING TEGEH..BALINESE MUSICAL SYMBOL COMBINING GONG
		(0x1B80 <= code && code <= 0x1B81) || // Mn   [2] SUNDANESE SIGN PANYECEK..SUNDANESE SIGN PANGLAYAR
		(0x1BA2 <= code && code <= 0x1BA5) || // Mn   [4] SUNDANESE CONSONANT SIGN PANYAKRA..SUNDANESE VOWEL SIGN PANYUKU
		(0x1BA8 <= code && code <= 0x1BA9) || // Mn   [2] SUNDANESE VOWEL SIGN PAMEPET..SUNDANESE VOWEL SIGN PANEULEUNG
		(0x1BAB <= code && code <= 0x1BAD) || // Mn   [3] SUNDANESE SIGN VIRAMA..SUNDANESE CONSONANT SIGN PASANGAN WA
		0x1BE6 == code || // Mn       BATAK SIGN TOMPI
		(0x1BE8 <= code && code <= 0x1BE9) || // Mn   [2] BATAK VOWEL SIGN PAKPAK E..BATAK VOWEL SIGN EE
		0x1BED == code || // Mn       BATAK VOWEL SIGN KARO O
		(0x1BEF <= code && code <= 0x1BF1) || // Mn   [3] BATAK VOWEL SIGN U FOR SIMALUNGUN SA..BATAK CONSONANT SIGN H
		(0x1C2C <= code && code <= 0x1C33) || // Mn   [8] LEPCHA VOWEL SIGN E..LEPCHA CONSONANT SIGN T
		(0x1C36 <= code && code <= 0x1C37) || // Mn   [2] LEPCHA SIGN RAN..LEPCHA SIGN NUKTA
		(0x1CD0 <= code && code <= 0x1CD2) || // Mn   [3] VEDIC TONE KARSHANA..VEDIC TONE PRENKHA
		(0x1CD4 <= code && code <= 0x1CE0) || // Mn  [13] VEDIC SIGN YAJURVEDIC MIDLINE SVARITA..VEDIC TONE RIGVEDIC KASHMIRI INDEPENDENT SVARITA
		(0x1CE2 <= code && code <= 0x1CE8) || // Mn   [7] VEDIC SIGN VISARGA SVARITA..VEDIC SIGN VISARGA ANUDATTA WITH TAIL
		0x1CED == code || // Mn       VEDIC SIGN TIRYAK
		0x1CF4 == code || // Mn       VEDIC TONE CANDRA ABOVE
		(0x1CF8 <= code && code <= 0x1CF9) || // Mn   [2] VEDIC TONE RING ABOVE..VEDIC TONE DOUBLE RING ABOVE
		(0x1DC0 <= code && code <= 0x1DF9) || // Mn  [58] COMBINING DOTTED GRAVE ACCENT..COMBINING WIDE INVERTED BRIDGE BELOW
		(0x1DFB <= code && code <= 0x1DFF) || // Mn   [5] COMBINING DELETION MARK..COMBINING RIGHT ARROWHEAD AND DOWN ARROWHEAD BELOW
		0x200C == code || // Cf       ZERO WIDTH NON-JOINER
		(0x20D0 <= code && code <= 0x20DC) || // Mn  [13] COMBINING LEFT HARPOON ABOVE..COMBINING FOUR DOTS ABOVE
		(0x20DD <= code && code <= 0x20E0) || // Me   [4] COMBINING ENCLOSING CIRCLE..COMBINING ENCLOSING CIRCLE BACKSLASH
		0x20E1 == code || // Mn       COMBINING LEFT RIGHT ARROW ABOVE
		(0x20E2 <= code && code <= 0x20E4) || // Me   [3] COMBINING ENCLOSING SCREEN..COMBINING ENCLOSING UPWARD POINTING TRIANGLE
		(0x20E5 <= code && code <= 0x20F0) || // Mn  [12] COMBINING REVERSE SOLIDUS OVERLAY..COMBINING ASTERISK ABOVE
		(0x2CEF <= code && code <= 0x2CF1) || // Mn   [3] COPTIC COMBINING NI ABOVE..COPTIC COMBINING SPIRITUS LENIS
		0x2D7F == code || // Mn       TIFINAGH CONSONANT JOINER
		(0x2DE0 <= code && code <= 0x2DFF) || // Mn  [32] COMBINING CYRILLIC LETTER BE..COMBINING CYRILLIC LETTER IOTIFIED BIG YUS
		(0x302A <= code && code <= 0x302D) || // Mn   [4] IDEOGRAPHIC LEVEL TONE MARK..IDEOGRAPHIC ENTERING TONE MARK
		(0x302E <= code && code <= 0x302F) || // Mc   [2] HANGUL SINGLE DOT TONE MARK..HANGUL DOUBLE DOT TONE MARK
		(0x3099 <= code && code <= 0x309A) || // Mn   [2] COMBINING KATAKANA-HIRAGANA VOICED SOUND MARK..COMBINING KATAKANA-HIRAGANA SEMI-VOICED SOUND MARK
		0xA66F == code || // Mn       COMBINING CYRILLIC VZMET
		(0xA670 <= code && code <= 0xA672) || // Me   [3] COMBINING CYRILLIC TEN MILLIONS SIGN..COMBINING CYRILLIC THOUSAND MILLIONS SIGN
		(0xA674 <= code && code <= 0xA67D) || // Mn  [10] COMBINING CYRILLIC LETTER UKRAINIAN IE..COMBINING CYRILLIC PAYEROK
		(0xA69E <= code && code <= 0xA69F) || // Mn   [2] COMBINING CYRILLIC LETTER EF..COMBINING CYRILLIC LETTER IOTIFIED E
		(0xA6F0 <= code && code <= 0xA6F1) || // Mn   [2] BAMUM COMBINING MARK KOQNDON..BAMUM COMBINING MARK TUKWENTIS
		0xA802 == code || // Mn       SYLOTI NAGRI SIGN DVISVARA
		0xA806 == code || // Mn       SYLOTI NAGRI SIGN HASANTA
		0xA80B == code || // Mn       SYLOTI NAGRI SIGN ANUSVARA
		(0xA825 <= code && code <= 0xA826) || // Mn   [2] SYLOTI NAGRI VOWEL SIGN U..SYLOTI NAGRI VOWEL SIGN E
		(0xA8C4 <= code && code <= 0xA8C5) || // Mn   [2] SAURASHTRA SIGN VIRAMA..SAURASHTRA SIGN CANDRABINDU
		(0xA8E0 <= code && code <= 0xA8F1) || // Mn  [18] COMBINING DEVANAGARI DIGIT ZERO..COMBINING DEVANAGARI SIGN AVAGRAHA
		(0xA926 <= code && code <= 0xA92D) || // Mn   [8] KAYAH LI VOWEL UE..KAYAH LI TONE CALYA PLOPHU
		(0xA947 <= code && code <= 0xA951) || // Mn  [11] REJANG VOWEL SIGN I..REJANG CONSONANT SIGN R
		(0xA980 <= code && code <= 0xA982) || // Mn   [3] JAVANESE SIGN PANYANGGA..JAVANESE SIGN LAYAR
		0xA9B3 == code || // Mn       JAVANESE SIGN CECAK TELU
		(0xA9B6 <= code && code <= 0xA9B9) || // Mn   [4] JAVANESE VOWEL SIGN WULU..JAVANESE VOWEL SIGN SUKU MENDUT
		0xA9BC == code || // Mn       JAVANESE VOWEL SIGN PEPET
		0xA9E5 == code || // Mn       MYANMAR SIGN SHAN SAW
		(0xAA29 <= code && code <= 0xAA2E) || // Mn   [6] CHAM VOWEL SIGN AA..CHAM VOWEL SIGN OE
		(0xAA31 <= code && code <= 0xAA32) || // Mn   [2] CHAM VOWEL SIGN AU..CHAM VOWEL SIGN UE
		(0xAA35 <= code && code <= 0xAA36) || // Mn   [2] CHAM CONSONANT SIGN LA..CHAM CONSONANT SIGN WA
		0xAA43 == code || // Mn       CHAM CONSONANT SIGN FINAL NG
		0xAA4C == code || // Mn       CHAM CONSONANT SIGN FINAL M
		0xAA7C == code || // Mn       MYANMAR SIGN TAI LAING TONE-2
		0xAAB0 == code || // Mn       TAI VIET MAI KANG
		(0xAAB2 <= code && code <= 0xAAB4) || // Mn   [3] TAI VIET VOWEL I..TAI VIET VOWEL U
		(0xAAB7 <= code && code <= 0xAAB8) || // Mn   [2] TAI VIET MAI KHIT..TAI VIET VOWEL IA
		(0xAABE <= code && code <= 0xAABF) || // Mn   [2] TAI VIET VOWEL AM..TAI VIET TONE MAI EK
		0xAAC1 == code || // Mn       TAI VIET TONE MAI THO
		(0xAAEC <= code && code <= 0xAAED) || // Mn   [2] MEETEI MAYEK VOWEL SIGN UU..MEETEI MAYEK VOWEL SIGN AAI
		0xAAF6 == code || // Mn       MEETEI MAYEK VIRAMA
		0xABE5 == code || // Mn       MEETEI MAYEK VOWEL SIGN ANAP
		0xABE8 == code || // Mn       MEETEI MAYEK VOWEL SIGN UNAP
		0xABED == code || // Mn       MEETEI MAYEK APUN IYEK
		0xFB1E == code || // Mn       HEBREW POINT JUDEO-SPANISH VARIKA
		(0xFE00 <= code && code <= 0xFE0F) || // Mn  [16] VARIATION SELECTOR-1..VARIATION SELECTOR-16
		(0xFE20 <= code && code <= 0xFE2F) || // Mn  [16] COMBINING LIGATURE LEFT HALF..COMBINING CYRILLIC TITLO RIGHT HALF
		(0xFF9E <= code && code <= 0xFF9F) || // Lm   [2] HALFWIDTH KATAKANA VOICED SOUND MARK..HALFWIDTH KATAKANA SEMI-VOICED SOUND MARK
		0x101FD == code || // Mn       PHAISTOS DISC SIGN COMBINING OBLIQUE STROKE
		0x102E0 == code || // Mn       COPTIC EPACT THOUSANDS MARK
		(0x10376 <= code && code <= 0x1037A) || // Mn   [5] COMBINING OLD PERMIC LETTER AN..COMBINING OLD PERMIC LETTER SII
		(0x10A01 <= code && code <= 0x10A03) || // Mn   [3] KHAROSHTHI VOWEL SIGN I..KHAROSHTHI VOWEL SIGN VOCALIC R
		(0x10A05 <= code && code <= 0x10A06) || // Mn   [2] KHAROSHTHI VOWEL SIGN E..KHAROSHTHI VOWEL SIGN O
		(0x10A0C <= code && code <= 0x10A0F) || // Mn   [4] KHAROSHTHI VOWEL LENGTH MARK..KHAROSHTHI SIGN VISARGA
		(0x10A38 <= code && code <= 0x10A3A) || // Mn   [3] KHAROSHTHI SIGN BAR ABOVE..KHAROSHTHI SIGN DOT BELOW
		0x10A3F == code || // Mn       KHAROSHTHI VIRAMA
		(0x10AE5 <= code && code <= 0x10AE6) || // Mn   [2] MANICHAEAN ABBREVIATION MARK ABOVE..MANICHAEAN ABBREVIATION MARK BELOW
		0x11001 == code || // Mn       BRAHMI SIGN ANUSVARA
		(0x11038 <= code && code <= 0x11046) || // Mn  [15] BRAHMI VOWEL SIGN AA..BRAHMI VIRAMA
		(0x1107F <= code && code <= 0x11081) || // Mn   [3] BRAHMI NUMBER JOINER..KAITHI SIGN ANUSVARA
		(0x110B3 <= code && code <= 0x110B6) || // Mn   [4] KAITHI VOWEL SIGN U..KAITHI VOWEL SIGN AI
		(0x110B9 <= code && code <= 0x110BA) || // Mn   [2] KAITHI SIGN VIRAMA..KAITHI SIGN NUKTA
		(0x11100 <= code && code <= 0x11102) || // Mn   [3] CHAKMA SIGN CANDRABINDU..CHAKMA SIGN VISARGA
		(0x11127 <= code && code <= 0x1112B) || // Mn   [5] CHAKMA VOWEL SIGN A..CHAKMA VOWEL SIGN UU
		(0x1112D <= code && code <= 0x11134) || // Mn   [8] CHAKMA VOWEL SIGN AI..CHAKMA MAAYYAA
		0x11173 == code || // Mn       MAHAJANI SIGN NUKTA
		(0x11180 <= code && code <= 0x11181) || // Mn   [2] SHARADA SIGN CANDRABINDU..SHARADA SIGN ANUSVARA
		(0x111B6 <= code && code <= 0x111BE) || // Mn   [9] SHARADA VOWEL SIGN U..SHARADA VOWEL SIGN O
		(0x111CA <= code && code <= 0x111CC) || // Mn   [3] SHARADA SIGN NUKTA..SHARADA EXTRA SHORT VOWEL MARK
		(0x1122F <= code && code <= 0x11231) || // Mn   [3] KHOJKI VOWEL SIGN U..KHOJKI VOWEL SIGN AI
		0x11234 == code || // Mn       KHOJKI SIGN ANUSVARA
		(0x11236 <= code && code <= 0x11237) || // Mn   [2] KHOJKI SIGN NUKTA..KHOJKI SIGN SHADDA
		0x1123E == code || // Mn       KHOJKI SIGN SUKUN
		0x112DF == code || // Mn       KHUDAWADI SIGN ANUSVARA
		(0x112E3 <= code && code <= 0x112EA) || // Mn   [8] KHUDAWADI VOWEL SIGN U..KHUDAWADI SIGN VIRAMA
		(0x11300 <= code && code <= 0x11301) || // Mn   [2] GRANTHA SIGN COMBINING ANUSVARA ABOVE..GRANTHA SIGN CANDRABINDU
		0x1133C == code || // Mn       GRANTHA SIGN NUKTA
		0x1133E == code || // Mc       GRANTHA VOWEL SIGN AA
		0x11340 == code || // Mn       GRANTHA VOWEL SIGN II
		0x11357 == code || // Mc       GRANTHA AU LENGTH MARK
		(0x11366 <= code && code <= 0x1136C) || // Mn   [7] COMBINING GRANTHA DIGIT ZERO..COMBINING GRANTHA DIGIT SIX
		(0x11370 <= code && code <= 0x11374) || // Mn   [5] COMBINING GRANTHA LETTER A..COMBINING GRANTHA LETTER PA
		(0x11438 <= code && code <= 0x1143F) || // Mn   [8] NEWA VOWEL SIGN U..NEWA VOWEL SIGN AI
		(0x11442 <= code && code <= 0x11444) || // Mn   [3] NEWA SIGN VIRAMA..NEWA SIGN ANUSVARA
		0x11446 == code || // Mn       NEWA SIGN NUKTA
		0x114B0 == code || // Mc       TIRHUTA VOWEL SIGN AA
		(0x114B3 <= code && code <= 0x114B8) || // Mn   [6] TIRHUTA VOWEL SIGN U..TIRHUTA VOWEL SIGN VOCALIC LL
		0x114BA == code || // Mn       TIRHUTA VOWEL SIGN SHORT E
		0x114BD == code || // Mc       TIRHUTA VOWEL SIGN SHORT O
		(0x114BF <= code && code <= 0x114C0) || // Mn   [2] TIRHUTA SIGN CANDRABINDU..TIRHUTA SIGN ANUSVARA
		(0x114C2 <= code && code <= 0x114C3) || // Mn   [2] TIRHUTA SIGN VIRAMA..TIRHUTA SIGN NUKTA
		0x115AF == code || // Mc       SIDDHAM VOWEL SIGN AA
		(0x115B2 <= code && code <= 0x115B5) || // Mn   [4] SIDDHAM VOWEL SIGN U..SIDDHAM VOWEL SIGN VOCALIC RR
		(0x115BC <= code && code <= 0x115BD) || // Mn   [2] SIDDHAM SIGN CANDRABINDU..SIDDHAM SIGN ANUSVARA
		(0x115BF <= code && code <= 0x115C0) || // Mn   [2] SIDDHAM SIGN VIRAMA..SIDDHAM SIGN NUKTA
		(0x115DC <= code && code <= 0x115DD) || // Mn   [2] SIDDHAM VOWEL SIGN ALTERNATE U..SIDDHAM VOWEL SIGN ALTERNATE UU
		(0x11633 <= code && code <= 0x1163A) || // Mn   [8] MODI VOWEL SIGN U..MODI VOWEL SIGN AI
		0x1163D == code || // Mn       MODI SIGN ANUSVARA
		(0x1163F <= code && code <= 0x11640) || // Mn   [2] MODI SIGN VIRAMA..MODI SIGN ARDHACANDRA
		0x116AB == code || // Mn       TAKRI SIGN ANUSVARA
		0x116AD == code || // Mn       TAKRI VOWEL SIGN AA
		(0x116B0 <= code && code <= 0x116B5) || // Mn   [6] TAKRI VOWEL SIGN U..TAKRI VOWEL SIGN AU
		0x116B7 == code || // Mn       TAKRI SIGN NUKTA
		(0x1171D <= code && code <= 0x1171F) || // Mn   [3] AHOM CONSONANT SIGN MEDIAL LA..AHOM CONSONANT SIGN MEDIAL LIGATING RA
		(0x11722 <= code && code <= 0x11725) || // Mn   [4] AHOM VOWEL SIGN I..AHOM VOWEL SIGN UU
		(0x11727 <= code && code <= 0x1172B) || // Mn   [5] AHOM VOWEL SIGN AW..AHOM SIGN KILLER
		(0x11A01 <= code && code <= 0x11A06) || // Mn   [6] ZANABAZAR SQUARE VOWEL SIGN I..ZANABAZAR SQUARE VOWEL SIGN O
		(0x11A09 <= code && code <= 0x11A0A) || // Mn   [2] ZANABAZAR SQUARE VOWEL SIGN REVERSED I..ZANABAZAR SQUARE VOWEL LENGTH MARK
		(0x11A33 <= code && code <= 0x11A38) || // Mn   [6] ZANABAZAR SQUARE FINAL CONSONANT MARK..ZANABAZAR SQUARE SIGN ANUSVARA
		(0x11A3B <= code && code <= 0x11A3E) || // Mn   [4] ZANABAZAR SQUARE CLUSTER-FINAL LETTER YA..ZANABAZAR SQUARE CLUSTER-FINAL LETTER VA
		0x11A47 == code || // Mn       ZANABAZAR SQUARE SUBJOINER
		(0x11A51 <= code && code <= 0x11A56) || // Mn   [6] SOYOMBO VOWEL SIGN I..SOYOMBO VOWEL SIGN OE
		(0x11A59 <= code && code <= 0x11A5B) || // Mn   [3] SOYOMBO VOWEL SIGN VOCALIC R..SOYOMBO VOWEL LENGTH MARK
		(0x11A8A <= code && code <= 0x11A96) || // Mn  [13] SOYOMBO FINAL CONSONANT SIGN G..SOYOMBO SIGN ANUSVARA
		(0x11A98 <= code && code <= 0x11A99) || // Mn   [2] SOYOMBO GEMINATION MARK..SOYOMBO SUBJOINER
		(0x11C30 <= code && code <= 0x11C36) || // Mn   [7] BHAIKSUKI VOWEL SIGN I..BHAIKSUKI VOWEL SIGN VOCALIC L
		(0x11C38 <= code && code <= 0x11C3D) || // Mn   [6] BHAIKSUKI VOWEL SIGN E..BHAIKSUKI SIGN ANUSVARA
		0x11C3F == code || // Mn       BHAIKSUKI SIGN VIRAMA
		(0x11C92 <= code && code <= 0x11CA7) || // Mn  [22] MARCHEN SUBJOINED LETTER KA..MARCHEN SUBJOINED LETTER ZA
		(0x11CAA <= code && code <= 0x11CB0) || // Mn   [7] MARCHEN SUBJOINED LETTER RA..MARCHEN VOWEL SIGN AA
		(0x11CB2 <= code && code <= 0x11CB3) || // Mn   [2] MARCHEN VOWEL SIGN U..MARCHEN VOWEL SIGN E
		(0x11CB5 <= code && code <= 0x11CB6) || // Mn   [2] MARCHEN SIGN ANUSVARA..MARCHEN SIGN CANDRABINDU
		(0x11D31 <= code && code <= 0x11D36) || // Mn   [6] MASARAM GONDI VOWEL SIGN AA..MASARAM GONDI VOWEL SIGN VOCALIC R
		0x11D3A == code || // Mn       MASARAM GONDI VOWEL SIGN E
		(0x11D3C <= code && code <= 0x11D3D) || // Mn   [2] MASARAM GONDI VOWEL SIGN AI..MASARAM GONDI VOWEL SIGN O
		(0x11D3F <= code && code <= 0x11D45) || // Mn   [7] MASARAM GONDI VOWEL SIGN AU..MASARAM GONDI VIRAMA
		0x11D47 == code || // Mn       MASARAM GONDI RA-KARA
		(0x16AF0 <= code && code <= 0x16AF4) || // Mn   [5] BASSA VAH COMBINING HIGH TONE..BASSA VAH COMBINING HIGH-LOW TONE
		(0x16B30 <= code && code <= 0x16B36) || // Mn   [7] PAHAWH HMONG MARK CIM TUB..PAHAWH HMONG MARK CIM TAUM
		(0x16F8F <= code && code <= 0x16F92) || // Mn   [4] MIAO TONE RIGHT..MIAO TONE BELOW
		(0x1BC9D <= code && code <= 0x1BC9E) || // Mn   [2] DUPLOYAN THICK LETTER SELECTOR..DUPLOYAN DOUBLE MARK
		0x1D165 == code || // Mc       MUSICAL SYMBOL COMBINING STEM
		(0x1D167 <= code && code <= 0x1D169) || // Mn   [3] MUSICAL SYMBOL COMBINING TREMOLO-1..MUSICAL SYMBOL COMBINING TREMOLO-3
		(0x1D16E <= code && code <= 0x1D172) || // Mc   [5] MUSICAL SYMBOL COMBINING FLAG-1..MUSICAL SYMBOL COMBINING FLAG-5
		(0x1D17B <= code && code <= 0x1D182) || // Mn   [8] MUSICAL SYMBOL COMBINING ACCENT..MUSICAL SYMBOL COMBINING LOURE
		(0x1D185 <= code && code <= 0x1D18B) || // Mn   [7] MUSICAL SYMBOL COMBINING DOIT..MUSICAL SYMBOL COMBINING TRIPLE TONGUE
		(0x1D1AA <= code && code <= 0x1D1AD) || // Mn   [4] MUSICAL SYMBOL COMBINING DOWN BOW..MUSICAL SYMBOL COMBINING SNAP PIZZICATO
		(0x1D242 <= code && code <= 0x1D244) || // Mn   [3] COMBINING GREEK MUSICAL TRISEME..COMBINING GREEK MUSICAL PENTASEME
		(0x1DA00 <= code && code <= 0x1DA36) || // Mn  [55] SIGNWRITING HEAD RIM..SIGNWRITING AIR SUCKING IN
		(0x1DA3B <= code && code <= 0x1DA6C) || // Mn  [50] SIGNWRITING MOUTH CLOSED NEUTRAL..SIGNWRITING EXCITEMENT
		0x1DA75 == code || // Mn       SIGNWRITING UPPER BODY TILTING FROM HIP JOINTS
		0x1DA84 == code || // Mn       SIGNWRITING LOCATION HEAD NECK
		(0x1DA9B <= code && code <= 0x1DA9F) || // Mn   [5] SIGNWRITING FILL MODIFIER-2..SIGNWRITING FILL MODIFIER-6
		(0x1DAA1 <= code && code <= 0x1DAAF) || // Mn  [15] SIGNWRITING ROTATION MODIFIER-2..SIGNWRITING ROTATION MODIFIER-16
		(0x1E000 <= code && code <= 0x1E006) || // Mn   [7] COMBINING GLAGOLITIC LETTER AZU..COMBINING GLAGOLITIC LETTER ZHIVETE
		(0x1E008 <= code && code <= 0x1E018) || // Mn  [17] COMBINING GLAGOLITIC LETTER ZEMLJA..COMBINING GLAGOLITIC LETTER HERU
		(0x1E01B <= code && code <= 0x1E021) || // Mn   [7] COMBINING GLAGOLITIC LETTER SHTA..COMBINING GLAGOLITIC LETTER YATI
		(0x1E023 <= code && code <= 0x1E024) || // Mn   [2] COMBINING GLAGOLITIC LETTER YU..COMBINING GLAGOLITIC LETTER SMALL YUS
		(0x1E026 <= code && code <= 0x1E02A) || // Mn   [5] COMBINING GLAGOLITIC LETTER YO..COMBINING GLAGOLITIC LETTER FITA
		(0x1E8D0 <= code && code <= 0x1E8D6) || // Mn   [7] MENDE KIKAKUI COMBINING NUMBER TEENS..MENDE KIKAKUI COMBINING NUMBER MILLIONS
		(0x1E944 <= code && code <= 0x1E94A) || // Mn   [7] ADLAM ALIF LENGTHENER..ADLAM NUKTA
		(0xE0020 <= code && code <= 0xE007F) || // Cf  [96] TAG SPACE..CANCEL TAG
		(0xE0100 <= code && code <= 0xE01EF) // Mn [240] VARIATION SELECTOR-17..VARIATION SELECTOR-256
		){
			return Extend;
		}
		
		
		if(
		(0x1F1E6 <= code && code <= 0x1F1FF) // So  [26] REGIONAL INDICATOR SYMBOL LETTER A..REGIONAL INDICATOR SYMBOL LETTER Z
		){
			return Regional_Indicator;
		}
		
		if(
		0x0903 == code || // Mc       DEVANAGARI SIGN VISARGA
		0x093B == code || // Mc       DEVANAGARI VOWEL SIGN OOE
		(0x093E <= code && code <= 0x0940) || // Mc   [3] DEVANAGARI VOWEL SIGN AA..DEVANAGARI VOWEL SIGN II
		(0x0949 <= code && code <= 0x094C) || // Mc   [4] DEVANAGARI VOWEL SIGN CANDRA O..DEVANAGARI VOWEL SIGN AU
		(0x094E <= code && code <= 0x094F) || // Mc   [2] DEVANAGARI VOWEL SIGN PRISHTHAMATRA E..DEVANAGARI VOWEL SIGN AW
		(0x0982 <= code && code <= 0x0983) || // Mc   [2] BENGALI SIGN ANUSVARA..BENGALI SIGN VISARGA
		(0x09BF <= code && code <= 0x09C0) || // Mc   [2] BENGALI VOWEL SIGN I..BENGALI VOWEL SIGN II
		(0x09C7 <= code && code <= 0x09C8) || // Mc   [2] BENGALI VOWEL SIGN E..BENGALI VOWEL SIGN AI
		(0x09CB <= code && code <= 0x09CC) || // Mc   [2] BENGALI VOWEL SIGN O..BENGALI VOWEL SIGN AU
		0x0A03 == code || // Mc       GURMUKHI SIGN VISARGA
		(0x0A3E <= code && code <= 0x0A40) || // Mc   [3] GURMUKHI VOWEL SIGN AA..GURMUKHI VOWEL SIGN II
		0x0A83 == code || // Mc       GUJARATI SIGN VISARGA
		(0x0ABE <= code && code <= 0x0AC0) || // Mc   [3] GUJARATI VOWEL SIGN AA..GUJARATI VOWEL SIGN II
		0x0AC9 == code || // Mc       GUJARATI VOWEL SIGN CANDRA O
		(0x0ACB <= code && code <= 0x0ACC) || // Mc   [2] GUJARATI VOWEL SIGN O..GUJARATI VOWEL SIGN AU
		(0x0B02 <= code && code <= 0x0B03) || // Mc   [2] ORIYA SIGN ANUSVARA..ORIYA SIGN VISARGA
		0x0B40 == code || // Mc       ORIYA VOWEL SIGN II
		(0x0B47 <= code && code <= 0x0B48) || // Mc   [2] ORIYA VOWEL SIGN E..ORIYA VOWEL SIGN AI
		(0x0B4B <= code && code <= 0x0B4C) || // Mc   [2] ORIYA VOWEL SIGN O..ORIYA VOWEL SIGN AU
		0x0BBF == code || // Mc       TAMIL VOWEL SIGN I
		(0x0BC1 <= code && code <= 0x0BC2) || // Mc   [2] TAMIL VOWEL SIGN U..TAMIL VOWEL SIGN UU
		(0x0BC6 <= code && code <= 0x0BC8) || // Mc   [3] TAMIL VOWEL SIGN E..TAMIL VOWEL SIGN AI
		(0x0BCA <= code && code <= 0x0BCC) || // Mc   [3] TAMIL VOWEL SIGN O..TAMIL VOWEL SIGN AU
		(0x0C01 <= code && code <= 0x0C03) || // Mc   [3] TELUGU SIGN CANDRABINDU..TELUGU SIGN VISARGA
		(0x0C41 <= code && code <= 0x0C44) || // Mc   [4] TELUGU VOWEL SIGN U..TELUGU VOWEL SIGN VOCALIC RR
		(0x0C82 <= code && code <= 0x0C83) || // Mc   [2] KANNADA SIGN ANUSVARA..KANNADA SIGN VISARGA
		0x0CBE == code || // Mc       KANNADA VOWEL SIGN AA
		(0x0CC0 <= code && code <= 0x0CC1) || // Mc   [2] KANNADA VOWEL SIGN II..KANNADA VOWEL SIGN U
		(0x0CC3 <= code && code <= 0x0CC4) || // Mc   [2] KANNADA VOWEL SIGN VOCALIC R..KANNADA VOWEL SIGN VOCALIC RR
		(0x0CC7 <= code && code <= 0x0CC8) || // Mc   [2] KANNADA VOWEL SIGN EE..KANNADA VOWEL SIGN AI
		(0x0CCA <= code && code <= 0x0CCB) || // Mc   [2] KANNADA VOWEL SIGN O..KANNADA VOWEL SIGN OO
		(0x0D02 <= code && code <= 0x0D03) || // Mc   [2] MALAYALAM SIGN ANUSVARA..MALAYALAM SIGN VISARGA
		(0x0D3F <= code && code <= 0x0D40) || // Mc   [2] MALAYALAM VOWEL SIGN I..MALAYALAM VOWEL SIGN II
		(0x0D46 <= code && code <= 0x0D48) || // Mc   [3] MALAYALAM VOWEL SIGN E..MALAYALAM VOWEL SIGN AI
		(0x0D4A <= code && code <= 0x0D4C) || // Mc   [3] MALAYALAM VOWEL SIGN O..MALAYALAM VOWEL SIGN AU
		(0x0D82 <= code && code <= 0x0D83) || // Mc   [2] SINHALA SIGN ANUSVARAYA..SINHALA SIGN VISARGAYA
		(0x0DD0 <= code && code <= 0x0DD1) || // Mc   [2] SINHALA VOWEL SIGN KETTI AEDA-PILLA..SINHALA VOWEL SIGN DIGA AEDA-PILLA
		(0x0DD8 <= code && code <= 0x0DDE) || // Mc   [7] SINHALA VOWEL SIGN GAETTA-PILLA..SINHALA VOWEL SIGN KOMBUVA HAA GAYANUKITTA
		(0x0DF2 <= code && code <= 0x0DF3) || // Mc   [2] SINHALA VOWEL SIGN DIGA GAETTA-PILLA..SINHALA VOWEL SIGN DIGA GAYANUKITTA
		0x0E33 == code || // Lo       THAI CHARACTER SARA AM
		0x0EB3 == code || // Lo       LAO VOWEL SIGN AM
		(0x0F3E <= code && code <= 0x0F3F) || // Mc   [2] TIBETAN SIGN YAR TSHES..TIBETAN SIGN MAR TSHES
		0x0F7F == code || // Mc       TIBETAN SIGN RNAM BCAD
		0x1031 == code || // Mc       MYANMAR VOWEL SIGN E
		(0x103B <= code && code <= 0x103C) || // Mc   [2] MYANMAR CONSONANT SIGN MEDIAL YA..MYANMAR CONSONANT SIGN MEDIAL RA
		(0x1056 <= code && code <= 0x1057) || // Mc   [2] MYANMAR VOWEL SIGN VOCALIC R..MYANMAR VOWEL SIGN VOCALIC RR
		0x1084 == code || // Mc       MYANMAR VOWEL SIGN SHAN E
		0x17B6 == code || // Mc       KHMER VOWEL SIGN AA
		(0x17BE <= code && code <= 0x17C5) || // Mc   [8] KHMER VOWEL SIGN OE..KHMER VOWEL SIGN AU
		(0x17C7 <= code && code <= 0x17C8) || // Mc   [2] KHMER SIGN REAHMUK..KHMER SIGN YUUKALEAPINTU
		(0x1923 <= code && code <= 0x1926) || // Mc   [4] LIMBU VOWEL SIGN EE..LIMBU VOWEL SIGN AU
		(0x1929 <= code && code <= 0x192B) || // Mc   [3] LIMBU SUBJOINED LETTER YA..LIMBU SUBJOINED LETTER WA
		(0x1930 <= code && code <= 0x1931) || // Mc   [2] LIMBU SMALL LETTER KA..LIMBU SMALL LETTER NGA
		(0x1933 <= code && code <= 0x1938) || // Mc   [6] LIMBU SMALL LETTER TA..LIMBU SMALL LETTER LA
		(0x1A19 <= code && code <= 0x1A1A) || // Mc   [2] BUGINESE VOWEL SIGN E..BUGINESE VOWEL SIGN O
		0x1A55 == code || // Mc       TAI THAM CONSONANT SIGN MEDIAL RA
		0x1A57 == code || // Mc       TAI THAM CONSONANT SIGN LA TANG LAI
		(0x1A6D <= code && code <= 0x1A72) || // Mc   [6] TAI THAM VOWEL SIGN OY..TAI THAM VOWEL SIGN THAM AI
		0x1B04 == code || // Mc       BALINESE SIGN BISAH
		0x1B35 == code || // Mc       BALINESE VOWEL SIGN TEDUNG
		0x1B3B == code || // Mc       BALINESE VOWEL SIGN RA REPA TEDUNG
		(0x1B3D <= code && code <= 0x1B41) || // Mc   [5] BALINESE VOWEL SIGN LA LENGA TEDUNG..BALINESE VOWEL SIGN TALING REPA TEDUNG
		(0x1B43 <= code && code <= 0x1B44) || // Mc   [2] BALINESE VOWEL SIGN PEPET TEDUNG..BALINESE ADEG ADEG
		0x1B82 == code || // Mc       SUNDANESE SIGN PANGWISAD
		0x1BA1 == code || // Mc       SUNDANESE CONSONANT SIGN PAMINGKAL
		(0x1BA6 <= code && code <= 0x1BA7) || // Mc   [2] SUNDANESE VOWEL SIGN PANAELAENG..SUNDANESE VOWEL SIGN PANOLONG
		0x1BAA == code || // Mc       SUNDANESE SIGN PAMAAEH
		0x1BE7 == code || // Mc       BATAK VOWEL SIGN E
		(0x1BEA <= code && code <= 0x1BEC) || // Mc   [3] BATAK VOWEL SIGN I..BATAK VOWEL SIGN O
		0x1BEE == code || // Mc       BATAK VOWEL SIGN U
		(0x1BF2 <= code && code <= 0x1BF3) || // Mc   [2] BATAK PANGOLAT..BATAK PANONGONAN
		(0x1C24 <= code && code <= 0x1C2B) || // Mc   [8] LEPCHA SUBJOINED LETTER YA..LEPCHA VOWEL SIGN UU
		(0x1C34 <= code && code <= 0x1C35) || // Mc   [2] LEPCHA CONSONANT SIGN NYIN-DO..LEPCHA CONSONANT SIGN KANG
		0x1CE1 == code || // Mc       VEDIC TONE ATHARVAVEDIC INDEPENDENT SVARITA
		(0x1CF2 <= code && code <= 0x1CF3) || // Mc   [2] VEDIC SIGN ARDHAVISARGA..VEDIC SIGN ROTATED ARDHAVISARGA
		0x1CF7 == code || // Mc       VEDIC SIGN ATIKRAMA
		(0xA823 <= code && code <= 0xA824) || // Mc   [2] SYLOTI NAGRI VOWEL SIGN A..SYLOTI NAGRI VOWEL SIGN I
		0xA827 == code || // Mc       SYLOTI NAGRI VOWEL SIGN OO
		(0xA880 <= code && code <= 0xA881) || // Mc   [2] SAURASHTRA SIGN ANUSVARA..SAURASHTRA SIGN VISARGA
		(0xA8B4 <= code && code <= 0xA8C3) || // Mc  [16] SAURASHTRA CONSONANT SIGN HAARU..SAURASHTRA VOWEL SIGN AU
		(0xA952 <= code && code <= 0xA953) || // Mc   [2] REJANG CONSONANT SIGN H..REJANG VIRAMA
		0xA983 == code || // Mc       JAVANESE SIGN WIGNYAN
		(0xA9B4 <= code && code <= 0xA9B5) || // Mc   [2] JAVANESE VOWEL SIGN TARUNG..JAVANESE VOWEL SIGN TOLONG
		(0xA9BA <= code && code <= 0xA9BB) || // Mc   [2] JAVANESE VOWEL SIGN TALING..JAVANESE VOWEL SIGN DIRGA MURE
		(0xA9BD <= code && code <= 0xA9C0) || // Mc   [4] JAVANESE CONSONANT SIGN KERET..JAVANESE PANGKON
		(0xAA2F <= code && code <= 0xAA30) || // Mc   [2] CHAM VOWEL SIGN O..CHAM VOWEL SIGN AI
		(0xAA33 <= code && code <= 0xAA34) || // Mc   [2] CHAM CONSONANT SIGN YA..CHAM CONSONANT SIGN RA
		0xAA4D == code || // Mc       CHAM CONSONANT SIGN FINAL H
		0xAAEB == code || // Mc       MEETEI MAYEK VOWEL SIGN II
		(0xAAEE <= code && code <= 0xAAEF) || // Mc   [2] MEETEI MAYEK VOWEL SIGN AU..MEETEI MAYEK VOWEL SIGN AAU
		0xAAF5 == code || // Mc       MEETEI MAYEK VOWEL SIGN VISARGA
		(0xABE3 <= code && code <= 0xABE4) || // Mc   [2] MEETEI MAYEK VOWEL SIGN ONAP..MEETEI MAYEK VOWEL SIGN INAP
		(0xABE6 <= code && code <= 0xABE7) || // Mc   [2] MEETEI MAYEK VOWEL SIGN YENAP..MEETEI MAYEK VOWEL SIGN SOUNAP
		(0xABE9 <= code && code <= 0xABEA) || // Mc   [2] MEETEI MAYEK VOWEL SIGN CHEINAP..MEETEI MAYEK VOWEL SIGN NUNG
		0xABEC == code || // Mc       MEETEI MAYEK LUM IYEK
		0x11000 == code || // Mc       BRAHMI SIGN CANDRABINDU
		0x11002 == code || // Mc       BRAHMI SIGN VISARGA
		0x11082 == code || // Mc       KAITHI SIGN VISARGA
		(0x110B0 <= code && code <= 0x110B2) || // Mc   [3] KAITHI VOWEL SIGN AA..KAITHI VOWEL SIGN II
		(0x110B7 <= code && code <= 0x110B8) || // Mc   [2] KAITHI VOWEL SIGN O..KAITHI VOWEL SIGN AU
		0x1112C == code || // Mc       CHAKMA VOWEL SIGN E
		0x11182 == code || // Mc       SHARADA SIGN VISARGA
		(0x111B3 <= code && code <= 0x111B5) || // Mc   [3] SHARADA VOWEL SIGN AA..SHARADA VOWEL SIGN II
		(0x111BF <= code && code <= 0x111C0) || // Mc   [2] SHARADA VOWEL SIGN AU..SHARADA SIGN VIRAMA
		(0x1122C <= code && code <= 0x1122E) || // Mc   [3] KHOJKI VOWEL SIGN AA..KHOJKI VOWEL SIGN II
		(0x11232 <= code && code <= 0x11233) || // Mc   [2] KHOJKI VOWEL SIGN O..KHOJKI VOWEL SIGN AU
		0x11235 == code || // Mc       KHOJKI SIGN VIRAMA
		(0x112E0 <= code && code <= 0x112E2) || // Mc   [3] KHUDAWADI VOWEL SIGN AA..KHUDAWADI VOWEL SIGN II
		(0x11302 <= code && code <= 0x11303) || // Mc   [2] GRANTHA SIGN ANUSVARA..GRANTHA SIGN VISARGA
		0x1133F == code || // Mc       GRANTHA VOWEL SIGN I
		(0x11341 <= code && code <= 0x11344) || // Mc   [4] GRANTHA VOWEL SIGN U..GRANTHA VOWEL SIGN VOCALIC RR
		(0x11347 <= code && code <= 0x11348) || // Mc   [2] GRANTHA VOWEL SIGN EE..GRANTHA VOWEL SIGN AI
		(0x1134B <= code && code <= 0x1134D) || // Mc   [3] GRANTHA VOWEL SIGN OO..GRANTHA SIGN VIRAMA
		(0x11362 <= code && code <= 0x11363) || // Mc   [2] GRANTHA VOWEL SIGN VOCALIC L..GRANTHA VOWEL SIGN VOCALIC LL
		(0x11435 <= code && code <= 0x11437) || // Mc   [3] NEWA VOWEL SIGN AA..NEWA VOWEL SIGN II
		(0x11440 <= code && code <= 0x11441) || // Mc   [2] NEWA VOWEL SIGN O..NEWA VOWEL SIGN AU
		0x11445 == code || // Mc       NEWA SIGN VISARGA
		(0x114B1 <= code && code <= 0x114B2) || // Mc   [2] TIRHUTA VOWEL SIGN I..TIRHUTA VOWEL SIGN II
		0x114B9 == code || // Mc       TIRHUTA VOWEL SIGN E
		(0x114BB <= code && code <= 0x114BC) || // Mc   [2] TIRHUTA VOWEL SIGN AI..TIRHUTA VOWEL SIGN O
		0x114BE == code || // Mc       TIRHUTA VOWEL SIGN AU
		0x114C1 == code || // Mc       TIRHUTA SIGN VISARGA
		(0x115B0 <= code && code <= 0x115B1) || // Mc   [2] SIDDHAM VOWEL SIGN I..SIDDHAM VOWEL SIGN II
		(0x115B8 <= code && code <= 0x115BB) || // Mc   [4] SIDDHAM VOWEL SIGN E..SIDDHAM VOWEL SIGN AU
		0x115BE == code || // Mc       SIDDHAM SIGN VISARGA
		(0x11630 <= code && code <= 0x11632) || // Mc   [3] MODI VOWEL SIGN AA..MODI VOWEL SIGN II
		(0x1163B <= code && code <= 0x1163C) || // Mc   [2] MODI VOWEL SIGN O..MODI VOWEL SIGN AU
		0x1163E == code || // Mc       MODI SIGN VISARGA
		0x116AC == code || // Mc       TAKRI SIGN VISARGA
		(0x116AE <= code && code <= 0x116AF) || // Mc   [2] TAKRI VOWEL SIGN I..TAKRI VOWEL SIGN II
		0x116B6 == code || // Mc       TAKRI SIGN VIRAMA
		(0x11720 <= code && code <= 0x11721) || // Mc   [2] AHOM VOWEL SIGN A..AHOM VOWEL SIGN AA
		0x11726 == code || // Mc       AHOM VOWEL SIGN E
		(0x11A07 <= code && code <= 0x11A08) || // Mc   [2] ZANABAZAR SQUARE VOWEL SIGN AI..ZANABAZAR SQUARE VOWEL SIGN AU
		0x11A39 == code || // Mc       ZANABAZAR SQUARE SIGN VISARGA
		(0x11A57 <= code && code <= 0x11A58) || // Mc   [2] SOYOMBO VOWEL SIGN AI..SOYOMBO VOWEL SIGN AU
		0x11A97 == code || // Mc       SOYOMBO SIGN VISARGA
		0x11C2F == code || // Mc       BHAIKSUKI VOWEL SIGN AA
		0x11C3E == code || // Mc       BHAIKSUKI SIGN VISARGA
		0x11CA9 == code || // Mc       MARCHEN SUBJOINED LETTER YA
		0x11CB1 == code || // Mc       MARCHEN VOWEL SIGN I
		0x11CB4 == code || // Mc       MARCHEN VOWEL SIGN O
		(0x16F51 <= code && code <= 0x16F7E) || // Mc  [46] MIAO SIGN ASPIRATION..MIAO VOWEL SIGN NG
		0x1D166 == code || // Mc       MUSICAL SYMBOL COMBINING SPRECHGESANG STEM
		0x1D16D == code // Mc       MUSICAL SYMBOL COMBINING AUGMENTATION DOT
		){
			return SpacingMark;
		}
		
		
		if(
		(0x1100 <= code && code <= 0x115F) || // Lo  [96] HANGUL CHOSEONG KIYEOK..HANGUL CHOSEONG FILLER
		(0xA960 <= code && code <= 0xA97C) // Lo  [29] HANGUL CHOSEONG TIKEUT-MIEUM..HANGUL CHOSEONG SSANGYEORINHIEUH
		){
			return L;
		}
		
		if(
		(0x1160 <= code && code <= 0x11A7) || // Lo  [72] HANGUL JUNGSEONG FILLER..HANGUL JUNGSEONG O-YAE
		(0xD7B0 <= code && code <= 0xD7C6) // Lo  [23] HANGUL JUNGSEONG O-YEO..HANGUL JUNGSEONG ARAEA-E
		){
			return V;
		}
		
		
		if(
		(0x11A8 <= code && code <= 0x11FF) || // Lo  [88] HANGUL JONGSEONG KIYEOK..HANGUL JONGSEONG SSANGNIEUN
		(0xD7CB <= code && code <= 0xD7FB) // Lo  [49] HANGUL JONGSEONG NIEUN-RIEUL..HANGUL JONGSEONG PHIEUPH-THIEUTH
		){
			return T;
		}
		
		if(
		0xAC00 == code || // Lo       HANGUL SYLLABLE GA
		0xAC1C == code || // Lo       HANGUL SYLLABLE GAE
		0xAC38 == code || // Lo       HANGUL SYLLABLE GYA
		0xAC54 == code || // Lo       HANGUL SYLLABLE GYAE
		0xAC70 == code || // Lo       HANGUL SYLLABLE GEO
		0xAC8C == code || // Lo       HANGUL SYLLABLE GE
		0xACA8 == code || // Lo       HANGUL SYLLABLE GYEO
		0xACC4 == code || // Lo       HANGUL SYLLABLE GYE
		0xACE0 == code || // Lo       HANGUL SYLLABLE GO
		0xACFC == code || // Lo       HANGUL SYLLABLE GWA
		0xAD18 == code || // Lo       HANGUL SYLLABLE GWAE
		0xAD34 == code || // Lo       HANGUL SYLLABLE GOE
		0xAD50 == code || // Lo       HANGUL SYLLABLE GYO
		0xAD6C == code || // Lo       HANGUL SYLLABLE GU
		0xAD88 == code || // Lo       HANGUL SYLLABLE GWEO
		0xADA4 == code || // Lo       HANGUL SYLLABLE GWE
		0xADC0 == code || // Lo       HANGUL SYLLABLE GWI
		0xADDC == code || // Lo       HANGUL SYLLABLE GYU
		0xADF8 == code || // Lo       HANGUL SYLLABLE GEU
		0xAE14 == code || // Lo       HANGUL SYLLABLE GYI
		0xAE30 == code || // Lo       HANGUL SYLLABLE GI
		0xAE4C == code || // Lo       HANGUL SYLLABLE GGA
		0xAE68 == code || // Lo       HANGUL SYLLABLE GGAE
		0xAE84 == code || // Lo       HANGUL SYLLABLE GGYA
		0xAEA0 == code || // Lo       HANGUL SYLLABLE GGYAE
		0xAEBC == code || // Lo       HANGUL SYLLABLE GGEO
		0xAED8 == code || // Lo       HANGUL SYLLABLE GGE
		0xAEF4 == code || // Lo       HANGUL SYLLABLE GGYEO
		0xAF10 == code || // Lo       HANGUL SYLLABLE GGYE
		0xAF2C == code || // Lo       HANGUL SYLLABLE GGO
		0xAF48 == code || // Lo       HANGUL SYLLABLE GGWA
		0xAF64 == code || // Lo       HANGUL SYLLABLE GGWAE
		0xAF80 == code || // Lo       HANGUL SYLLABLE GGOE
		0xAF9C == code || // Lo       HANGUL SYLLABLE GGYO
		0xAFB8 == code || // Lo       HANGUL SYLLABLE GGU
		0xAFD4 == code || // Lo       HANGUL SYLLABLE GGWEO
		0xAFF0 == code || // Lo       HANGUL SYLLABLE GGWE
		0xB00C == code || // Lo       HANGUL SYLLABLE GGWI
		0xB028 == code || // Lo       HANGUL SYLLABLE GGYU
		0xB044 == code || // Lo       HANGUL SYLLABLE GGEU
		0xB060 == code || // Lo       HANGUL SYLLABLE GGYI
		0xB07C == code || // Lo       HANGUL SYLLABLE GGI
		0xB098 == code || // Lo       HANGUL SYLLABLE NA
		0xB0B4 == code || // Lo       HANGUL SYLLABLE NAE
		0xB0D0 == code || // Lo       HANGUL SYLLABLE NYA
		0xB0EC == code || // Lo       HANGUL SYLLABLE NYAE
		0xB108 == code || // Lo       HANGUL SYLLABLE NEO
		0xB124 == code || // Lo       HANGUL SYLLABLE NE
		0xB140 == code || // Lo       HANGUL SYLLABLE NYEO
		0xB15C == code || // Lo       HANGUL SYLLABLE NYE
		0xB178 == code || // Lo       HANGUL SYLLABLE NO
		0xB194 == code || // Lo       HANGUL SYLLABLE NWA
		0xB1B0 == code || // Lo       HANGUL SYLLABLE NWAE
		0xB1CC == code || // Lo       HANGUL SYLLABLE NOE
		0xB1E8 == code || // Lo       HANGUL SYLLABLE NYO
		0xB204 == code || // Lo       HANGUL SYLLABLE NU
		0xB220 == code || // Lo       HANGUL SYLLABLE NWEO
		0xB23C == code || // Lo       HANGUL SYLLABLE NWE
		0xB258 == code || // Lo       HANGUL SYLLABLE NWI
		0xB274 == code || // Lo       HANGUL SYLLABLE NYU
		0xB290 == code || // Lo       HANGUL SYLLABLE NEU
		0xB2AC == code || // Lo       HANGUL SYLLABLE NYI
		0xB2C8 == code || // Lo       HANGUL SYLLABLE NI
		0xB2E4 == code || // Lo       HANGUL SYLLABLE DA
		0xB300 == code || // Lo       HANGUL SYLLABLE DAE
		0xB31C == code || // Lo       HANGUL SYLLABLE DYA
		0xB338 == code || // Lo       HANGUL SYLLABLE DYAE
		0xB354 == code || // Lo       HANGUL SYLLABLE DEO
		0xB370 == code || // Lo       HANGUL SYLLABLE DE
		0xB38C == code || // Lo       HANGUL SYLLABLE DYEO
		0xB3A8 == code || // Lo       HANGUL SYLLABLE DYE
		0xB3C4 == code || // Lo       HANGUL SYLLABLE DO
		0xB3E0 == code || // Lo       HANGUL SYLLABLE DWA
		0xB3FC == code || // Lo       HANGUL SYLLABLE DWAE
		0xB418 == code || // Lo       HANGUL SYLLABLE DOE
		0xB434 == code || // Lo       HANGUL SYLLABLE DYO
		0xB450 == code || // Lo       HANGUL SYLLABLE DU
		0xB46C == code || // Lo       HANGUL SYLLABLE DWEO
		0xB488 == code || // Lo       HANGUL SYLLABLE DWE
		0xB4A4 == code || // Lo       HANGUL SYLLABLE DWI
		0xB4C0 == code || // Lo       HANGUL SYLLABLE DYU
		0xB4DC == code || // Lo       HANGUL SYLLABLE DEU
		0xB4F8 == code || // Lo       HANGUL SYLLABLE DYI
		0xB514 == code || // Lo       HANGUL SYLLABLE DI
		0xB530 == code || // Lo       HANGUL SYLLABLE DDA
		0xB54C == code || // Lo       HANGUL SYLLABLE DDAE
		0xB568 == code || // Lo       HANGUL SYLLABLE DDYA
		0xB584 == code || // Lo       HANGUL SYLLABLE DDYAE
		0xB5A0 == code || // Lo       HANGUL SYLLABLE DDEO
		0xB5BC == code || // Lo       HANGUL SYLLABLE DDE
		0xB5D8 == code || // Lo       HANGUL SYLLABLE DDYEO
		0xB5F4 == code || // Lo       HANGUL SYLLABLE DDYE
		0xB610 == code || // Lo       HANGUL SYLLABLE DDO
		0xB62C == code || // Lo       HANGUL SYLLABLE DDWA
		0xB648 == code || // Lo       HANGUL SYLLABLE DDWAE
		0xB664 == code || // Lo       HANGUL SYLLABLE DDOE
		0xB680 == code || // Lo       HANGUL SYLLABLE DDYO
		0xB69C == code || // Lo       HANGUL SYLLABLE DDU
		0xB6B8 == code || // Lo       HANGUL SYLLABLE DDWEO
		0xB6D4 == code || // Lo       HANGUL SYLLABLE DDWE
		0xB6F0 == code || // Lo       HANGUL SYLLABLE DDWI
		0xB70C == code || // Lo       HANGUL SYLLABLE DDYU
		0xB728 == code || // Lo       HANGUL SYLLABLE DDEU
		0xB744 == code || // Lo       HANGUL SYLLABLE DDYI
		0xB760 == code || // Lo       HANGUL SYLLABLE DDI
		0xB77C == code || // Lo       HANGUL SYLLABLE RA
		0xB798 == code || // Lo       HANGUL SYLLABLE RAE
		0xB7B4 == code || // Lo       HANGUL SYLLABLE RYA
		0xB7D0 == code || // Lo       HANGUL SYLLABLE RYAE
		0xB7EC == code || // Lo       HANGUL SYLLABLE REO
		0xB808 == code || // Lo       HANGUL SYLLABLE RE
		0xB824 == code || // Lo       HANGUL SYLLABLE RYEO
		0xB840 == code || // Lo       HANGUL SYLLABLE RYE
		0xB85C == code || // Lo       HANGUL SYLLABLE RO
		0xB878 == code || // Lo       HANGUL SYLLABLE RWA
		0xB894 == code || // Lo       HANGUL SYLLABLE RWAE
		0xB8B0 == code || // Lo       HANGUL SYLLABLE ROE
		0xB8CC == code || // Lo       HANGUL SYLLABLE RYO
		0xB8E8 == code || // Lo       HANGUL SYLLABLE RU
		0xB904 == code || // Lo       HANGUL SYLLABLE RWEO
		0xB920 == code || // Lo       HANGUL SYLLABLE RWE
		0xB93C == code || // Lo       HANGUL SYLLABLE RWI
		0xB958 == code || // Lo       HANGUL SYLLABLE RYU
		0xB974 == code || // Lo       HANGUL SYLLABLE REU
		0xB990 == code || // Lo       HANGUL SYLLABLE RYI
		0xB9AC == code || // Lo       HANGUL SYLLABLE RI
		0xB9C8 == code || // Lo       HANGUL SYLLABLE MA
		0xB9E4 == code || // Lo       HANGUL SYLLABLE MAE
		0xBA00 == code || // Lo       HANGUL SYLLABLE MYA
		0xBA1C == code || // Lo       HANGUL SYLLABLE MYAE
		0xBA38 == code || // Lo       HANGUL SYLLABLE MEO
		0xBA54 == code || // Lo       HANGUL SYLLABLE ME
		0xBA70 == code || // Lo       HANGUL SYLLABLE MYEO
		0xBA8C == code || // Lo       HANGUL SYLLABLE MYE
		0xBAA8 == code || // Lo       HANGUL SYLLABLE MO
		0xBAC4 == code || // Lo       HANGUL SYLLABLE MWA
		0xBAE0 == code || // Lo       HANGUL SYLLABLE MWAE
		0xBAFC == code || // Lo       HANGUL SYLLABLE MOE
		0xBB18 == code || // Lo       HANGUL SYLLABLE MYO
		0xBB34 == code || // Lo       HANGUL SYLLABLE MU
		0xBB50 == code || // Lo       HANGUL SYLLABLE MWEO
		0xBB6C == code || // Lo       HANGUL SYLLABLE MWE
		0xBB88 == code || // Lo       HANGUL SYLLABLE MWI
		0xBBA4 == code || // Lo       HANGUL SYLLABLE MYU
		0xBBC0 == code || // Lo       HANGUL SYLLABLE MEU
		0xBBDC == code || // Lo       HANGUL SYLLABLE MYI
		0xBBF8 == code || // Lo       HANGUL SYLLABLE MI
		0xBC14 == code || // Lo       HANGUL SYLLABLE BA
		0xBC30 == code || // Lo       HANGUL SYLLABLE BAE
		0xBC4C == code || // Lo       HANGUL SYLLABLE BYA
		0xBC68 == code || // Lo       HANGUL SYLLABLE BYAE
		0xBC84 == code || // Lo       HANGUL SYLLABLE BEO
		0xBCA0 == code || // Lo       HANGUL SYLLABLE BE
		0xBCBC == code || // Lo       HANGUL SYLLABLE BYEO
		0xBCD8 == code || // Lo       HANGUL SYLLABLE BYE
		0xBCF4 == code || // Lo       HANGUL SYLLABLE BO
		0xBD10 == code || // Lo       HANGUL SYLLABLE BWA
		0xBD2C == code || // Lo       HANGUL SYLLABLE BWAE
		0xBD48 == code || // Lo       HANGUL SYLLABLE BOE
		0xBD64 == code || // Lo       HANGUL SYLLABLE BYO
		0xBD80 == code || // Lo       HANGUL SYLLABLE BU
		0xBD9C == code || // Lo       HANGUL SYLLABLE BWEO
		0xBDB8 == code || // Lo       HANGUL SYLLABLE BWE
		0xBDD4 == code || // Lo       HANGUL SYLLABLE BWI
		0xBDF0 == code || // Lo       HANGUL SYLLABLE BYU
		0xBE0C == code || // Lo       HANGUL SYLLABLE BEU
		0xBE28 == code || // Lo       HANGUL SYLLABLE BYI
		0xBE44 == code || // Lo       HANGUL SYLLABLE BI
		0xBE60 == code || // Lo       HANGUL SYLLABLE BBA
		0xBE7C == code || // Lo       HANGUL SYLLABLE BBAE
		0xBE98 == code || // Lo       HANGUL SYLLABLE BBYA
		0xBEB4 == code || // Lo       HANGUL SYLLABLE BBYAE
		0xBED0 == code || // Lo       HANGUL SYLLABLE BBEO
		0xBEEC == code || // Lo       HANGUL SYLLABLE BBE
		0xBF08 == code || // Lo       HANGUL SYLLABLE BBYEO
		0xBF24 == code || // Lo       HANGUL SYLLABLE BBYE
		0xBF40 == code || // Lo       HANGUL SYLLABLE BBO
		0xBF5C == code || // Lo       HANGUL SYLLABLE BBWA
		0xBF78 == code || // Lo       HANGUL SYLLABLE BBWAE
		0xBF94 == code || // Lo       HANGUL SYLLABLE BBOE
		0xBFB0 == code || // Lo       HANGUL SYLLABLE BBYO
		0xBFCC == code || // Lo       HANGUL SYLLABLE BBU
		0xBFE8 == code || // Lo       HANGUL SYLLABLE BBWEO
		0xC004 == code || // Lo       HANGUL SYLLABLE BBWE
		0xC020 == code || // Lo       HANGUL SYLLABLE BBWI
		0xC03C == code || // Lo       HANGUL SYLLABLE BBYU
		0xC058 == code || // Lo       HANGUL SYLLABLE BBEU
		0xC074 == code || // Lo       HANGUL SYLLABLE BBYI
		0xC090 == code || // Lo       HANGUL SYLLABLE BBI
		0xC0AC == code || // Lo       HANGUL SYLLABLE SA
		0xC0C8 == code || // Lo       HANGUL SYLLABLE SAE
		0xC0E4 == code || // Lo       HANGUL SYLLABLE SYA
		0xC100 == code || // Lo       HANGUL SYLLABLE SYAE
		0xC11C == code || // Lo       HANGUL SYLLABLE SEO
		0xC138 == code || // Lo       HANGUL SYLLABLE SE
		0xC154 == code || // Lo       HANGUL SYLLABLE SYEO
		0xC170 == code || // Lo       HANGUL SYLLABLE SYE
		0xC18C == code || // Lo       HANGUL SYLLABLE SO
		0xC1A8 == code || // Lo       HANGUL SYLLABLE SWA
		0xC1C4 == code || // Lo       HANGUL SYLLABLE SWAE
		0xC1E0 == code || // Lo       HANGUL SYLLABLE SOE
		0xC1FC == code || // Lo       HANGUL SYLLABLE SYO
		0xC218 == code || // Lo       HANGUL SYLLABLE SU
		0xC234 == code || // Lo       HANGUL SYLLABLE SWEO
		0xC250 == code || // Lo       HANGUL SYLLABLE SWE
		0xC26C == code || // Lo       HANGUL SYLLABLE SWI
		0xC288 == code || // Lo       HANGUL SYLLABLE SYU
		0xC2A4 == code || // Lo       HANGUL SYLLABLE SEU
		0xC2C0 == code || // Lo       HANGUL SYLLABLE SYI
		0xC2DC == code || // Lo       HANGUL SYLLABLE SI
		0xC2F8 == code || // Lo       HANGUL SYLLABLE SSA
		0xC314 == code || // Lo       HANGUL SYLLABLE SSAE
		0xC330 == code || // Lo       HANGUL SYLLABLE SSYA
		0xC34C == code || // Lo       HANGUL SYLLABLE SSYAE
		0xC368 == code || // Lo       HANGUL SYLLABLE SSEO
		0xC384 == code || // Lo       HANGUL SYLLABLE SSE
		0xC3A0 == code || // Lo       HANGUL SYLLABLE SSYEO
		0xC3BC == code || // Lo       HANGUL SYLLABLE SSYE
		0xC3D8 == code || // Lo       HANGUL SYLLABLE SSO
		0xC3F4 == code || // Lo       HANGUL SYLLABLE SSWA
		0xC410 == code || // Lo       HANGUL SYLLABLE SSWAE
		0xC42C == code || // Lo       HANGUL SYLLABLE SSOE
		0xC448 == code || // Lo       HANGUL SYLLABLE SSYO
		0xC464 == code || // Lo       HANGUL SYLLABLE SSU
		0xC480 == code || // Lo       HANGUL SYLLABLE SSWEO
		0xC49C == code || // Lo       HANGUL SYLLABLE SSWE
		0xC4B8 == code || // Lo       HANGUL SYLLABLE SSWI
		0xC4D4 == code || // Lo       HANGUL SYLLABLE SSYU
		0xC4F0 == code || // Lo       HANGUL SYLLABLE SSEU
		0xC50C == code || // Lo       HANGUL SYLLABLE SSYI
		0xC528 == code || // Lo       HANGUL SYLLABLE SSI
		0xC544 == code || // Lo       HANGUL SYLLABLE A
		0xC560 == code || // Lo       HANGUL SYLLABLE AE
		0xC57C == code || // Lo       HANGUL SYLLABLE YA
		0xC598 == code || // Lo       HANGUL SYLLABLE YAE
		0xC5B4 == code || // Lo       HANGUL SYLLABLE EO
		0xC5D0 == code || // Lo       HANGUL SYLLABLE E
		0xC5EC == code || // Lo       HANGUL SYLLABLE YEO
		0xC608 == code || // Lo       HANGUL SYLLABLE YE
		0xC624 == code || // Lo       HANGUL SYLLABLE O
		0xC640 == code || // Lo       HANGUL SYLLABLE WA
		0xC65C == code || // Lo       HANGUL SYLLABLE WAE
		0xC678 == code || // Lo       HANGUL SYLLABLE OE
		0xC694 == code || // Lo       HANGUL SYLLABLE YO
		0xC6B0 == code || // Lo       HANGUL SYLLABLE U
		0xC6CC == code || // Lo       HANGUL SYLLABLE WEO
		0xC6E8 == code || // Lo       HANGUL SYLLABLE WE
		0xC704 == code || // Lo       HANGUL SYLLABLE WI
		0xC720 == code || // Lo       HANGUL SYLLABLE YU
		0xC73C == code || // Lo       HANGUL SYLLABLE EU
		0xC758 == code || // Lo       HANGUL SYLLABLE YI
		0xC774 == code || // Lo       HANGUL SYLLABLE I
		0xC790 == code || // Lo       HANGUL SYLLABLE JA
		0xC7AC == code || // Lo       HANGUL SYLLABLE JAE
		0xC7C8 == code || // Lo       HANGUL SYLLABLE JYA
		0xC7E4 == code || // Lo       HANGUL SYLLABLE JYAE
		0xC800 == code || // Lo       HANGUL SYLLABLE JEO
		0xC81C == code || // Lo       HANGUL SYLLABLE JE
		0xC838 == code || // Lo       HANGUL SYLLABLE JYEO
		0xC854 == code || // Lo       HANGUL SYLLABLE JYE
		0xC870 == code || // Lo       HANGUL SYLLABLE JO
		0xC88C == code || // Lo       HANGUL SYLLABLE JWA
		0xC8A8 == code || // Lo       HANGUL SYLLABLE JWAE
		0xC8C4 == code || // Lo       HANGUL SYLLABLE JOE
		0xC8E0 == code || // Lo       HANGUL SYLLABLE JYO
		0xC8FC == code || // Lo       HANGUL SYLLABLE JU
		0xC918 == code || // Lo       HANGUL SYLLABLE JWEO
		0xC934 == code || // Lo       HANGUL SYLLABLE JWE
		0xC950 == code || // Lo       HANGUL SYLLABLE JWI
		0xC96C == code || // Lo       HANGUL SYLLABLE JYU
		0xC988 == code || // Lo       HANGUL SYLLABLE JEU
		0xC9A4 == code || // Lo       HANGUL SYLLABLE JYI
		0xC9C0 == code || // Lo       HANGUL SYLLABLE JI
		0xC9DC == code || // Lo       HANGUL SYLLABLE JJA
		0xC9F8 == code || // Lo       HANGUL SYLLABLE JJAE
		0xCA14 == code || // Lo       HANGUL SYLLABLE JJYA
		0xCA30 == code || // Lo       HANGUL SYLLABLE JJYAE
		0xCA4C == code || // Lo       HANGUL SYLLABLE JJEO
		0xCA68 == code || // Lo       HANGUL SYLLABLE JJE
		0xCA84 == code || // Lo       HANGUL SYLLABLE JJYEO
		0xCAA0 == code || // Lo       HANGUL SYLLABLE JJYE
		0xCABC == code || // Lo       HANGUL SYLLABLE JJO
		0xCAD8 == code || // Lo       HANGUL SYLLABLE JJWA
		0xCAF4 == code || // Lo       HANGUL SYLLABLE JJWAE
		0xCB10 == code || // Lo       HANGUL SYLLABLE JJOE
		0xCB2C == code || // Lo       HANGUL SYLLABLE JJYO
		0xCB48 == code || // Lo       HANGUL SYLLABLE JJU
		0xCB64 == code || // Lo       HANGUL SYLLABLE JJWEO
		0xCB80 == code || // Lo       HANGUL SYLLABLE JJWE
		0xCB9C == code || // Lo       HANGUL SYLLABLE JJWI
		0xCBB8 == code || // Lo       HANGUL SYLLABLE JJYU
		0xCBD4 == code || // Lo       HANGUL SYLLABLE JJEU
		0xCBF0 == code || // Lo       HANGUL SYLLABLE JJYI
		0xCC0C == code || // Lo       HANGUL SYLLABLE JJI
		0xCC28 == code || // Lo       HANGUL SYLLABLE CA
		0xCC44 == code || // Lo       HANGUL SYLLABLE CAE
		0xCC60 == code || // Lo       HANGUL SYLLABLE CYA
		0xCC7C == code || // Lo       HANGUL SYLLABLE CYAE
		0xCC98 == code || // Lo       HANGUL SYLLABLE CEO
		0xCCB4 == code || // Lo       HANGUL SYLLABLE CE
		0xCCD0 == code || // Lo       HANGUL SYLLABLE CYEO
		0xCCEC == code || // Lo       HANGUL SYLLABLE CYE
		0xCD08 == code || // Lo       HANGUL SYLLABLE CO
		0xCD24 == code || // Lo       HANGUL SYLLABLE CWA
		0xCD40 == code || // Lo       HANGUL SYLLABLE CWAE
		0xCD5C == code || // Lo       HANGUL SYLLABLE COE
		0xCD78 == code || // Lo       HANGUL SYLLABLE CYO
		0xCD94 == code || // Lo       HANGUL SYLLABLE CU
		0xCDB0 == code || // Lo       HANGUL SYLLABLE CWEO
		0xCDCC == code || // Lo       HANGUL SYLLABLE CWE
		0xCDE8 == code || // Lo       HANGUL SYLLABLE CWI
		0xCE04 == code || // Lo       HANGUL SYLLABLE CYU
		0xCE20 == code || // Lo       HANGUL SYLLABLE CEU
		0xCE3C == code || // Lo       HANGUL SYLLABLE CYI
		0xCE58 == code || // Lo       HANGUL SYLLABLE CI
		0xCE74 == code || // Lo       HANGUL SYLLABLE KA
		0xCE90 == code || // Lo       HANGUL SYLLABLE KAE
		0xCEAC == code || // Lo       HANGUL SYLLABLE KYA
		0xCEC8 == code || // Lo       HANGUL SYLLABLE KYAE
		0xCEE4 == code || // Lo       HANGUL SYLLABLE KEO
		0xCF00 == code || // Lo       HANGUL SYLLABLE KE
		0xCF1C == code || // Lo       HANGUL SYLLABLE KYEO
		0xCF38 == code || // Lo       HANGUL SYLLABLE KYE
		0xCF54 == code || // Lo       HANGUL SYLLABLE KO
		0xCF70 == code || // Lo       HANGUL SYLLABLE KWA
		0xCF8C == code || // Lo       HANGUL SYLLABLE KWAE
		0xCFA8 == code || // Lo       HANGUL SYLLABLE KOE
		0xCFC4 == code || // Lo       HANGUL SYLLABLE KYO
		0xCFE0 == code || // Lo       HANGUL SYLLABLE KU
		0xCFFC == code || // Lo       HANGUL SYLLABLE KWEO
		0xD018 == code || // Lo       HANGUL SYLLABLE KWE
		0xD034 == code || // Lo       HANGUL SYLLABLE KWI
		0xD050 == code || // Lo       HANGUL SYLLABLE KYU
		0xD06C == code || // Lo       HANGUL SYLLABLE KEU
		0xD088 == code || // Lo       HANGUL SYLLABLE KYI
		0xD0A4 == code || // Lo       HANGUL SYLLABLE KI
		0xD0C0 == code || // Lo       HANGUL SYLLABLE TA
		0xD0DC == code || // Lo       HANGUL SYLLABLE TAE
		0xD0F8 == code || // Lo       HANGUL SYLLABLE TYA
		0xD114 == code || // Lo       HANGUL SYLLABLE TYAE
		0xD130 == code || // Lo       HANGUL SYLLABLE TEO
		0xD14C == code || // Lo       HANGUL SYLLABLE TE
		0xD168 == code || // Lo       HANGUL SYLLABLE TYEO
		0xD184 == code || // Lo       HANGUL SYLLABLE TYE
		0xD1A0 == code || // Lo       HANGUL SYLLABLE TO
		0xD1BC == code || // Lo       HANGUL SYLLABLE TWA
		0xD1D8 == code || // Lo       HANGUL SYLLABLE TWAE
		0xD1F4 == code || // Lo       HANGUL SYLLABLE TOE
		0xD210 == code || // Lo       HANGUL SYLLABLE TYO
		0xD22C == code || // Lo       HANGUL SYLLABLE TU
		0xD248 == code || // Lo       HANGUL SYLLABLE TWEO
		0xD264 == code || // Lo       HANGUL SYLLABLE TWE
		0xD280 == code || // Lo       HANGUL SYLLABLE TWI
		0xD29C == code || // Lo       HANGUL SYLLABLE TYU
		0xD2B8 == code || // Lo       HANGUL SYLLABLE TEU
		0xD2D4 == code || // Lo       HANGUL SYLLABLE TYI
		0xD2F0 == code || // Lo       HANGUL SYLLABLE TI
		0xD30C == code || // Lo       HANGUL SYLLABLE PA
		0xD328 == code || // Lo       HANGUL SYLLABLE PAE
		0xD344 == code || // Lo       HANGUL SYLLABLE PYA
		0xD360 == code || // Lo       HANGUL SYLLABLE PYAE
		0xD37C == code || // Lo       HANGUL SYLLABLE PEO
		0xD398 == code || // Lo       HANGUL SYLLABLE PE
		0xD3B4 == code || // Lo       HANGUL SYLLABLE PYEO
		0xD3D0 == code || // Lo       HANGUL SYLLABLE PYE
		0xD3EC == code || // Lo       HANGUL SYLLABLE PO
		0xD408 == code || // Lo       HANGUL SYLLABLE PWA
		0xD424 == code || // Lo       HANGUL SYLLABLE PWAE
		0xD440 == code || // Lo       HANGUL SYLLABLE POE
		0xD45C == code || // Lo       HANGUL SYLLABLE PYO
		0xD478 == code || // Lo       HANGUL SYLLABLE PU
		0xD494 == code || // Lo       HANGUL SYLLABLE PWEO
		0xD4B0 == code || // Lo       HANGUL SYLLABLE PWE
		0xD4CC == code || // Lo       HANGUL SYLLABLE PWI
		0xD4E8 == code || // Lo       HANGUL SYLLABLE PYU
		0xD504 == code || // Lo       HANGUL SYLLABLE PEU
		0xD520 == code || // Lo       HANGUL SYLLABLE PYI
		0xD53C == code || // Lo       HANGUL SYLLABLE PI
		0xD558 == code || // Lo       HANGUL SYLLABLE HA
		0xD574 == code || // Lo       HANGUL SYLLABLE HAE
		0xD590 == code || // Lo       HANGUL SYLLABLE HYA
		0xD5AC == code || // Lo       HANGUL SYLLABLE HYAE
		0xD5C8 == code || // Lo       HANGUL SYLLABLE HEO
		0xD5E4 == code || // Lo       HANGUL SYLLABLE HE
		0xD600 == code || // Lo       HANGUL SYLLABLE HYEO
		0xD61C == code || // Lo       HANGUL SYLLABLE HYE
		0xD638 == code || // Lo       HANGUL SYLLABLE HO
		0xD654 == code || // Lo       HANGUL SYLLABLE HWA
		0xD670 == code || // Lo       HANGUL SYLLABLE HWAE
		0xD68C == code || // Lo       HANGUL SYLLABLE HOE
		0xD6A8 == code || // Lo       HANGUL SYLLABLE HYO
		0xD6C4 == code || // Lo       HANGUL SYLLABLE HU
		0xD6E0 == code || // Lo       HANGUL SYLLABLE HWEO
		0xD6FC == code || // Lo       HANGUL SYLLABLE HWE
		0xD718 == code || // Lo       HANGUL SYLLABLE HWI
		0xD734 == code || // Lo       HANGUL SYLLABLE HYU
		0xD750 == code || // Lo       HANGUL SYLLABLE HEU
		0xD76C == code || // Lo       HANGUL SYLLABLE HYI
		0xD788 == code // Lo       HANGUL SYLLABLE HI
		){
			return LV;
		}
		
		if(
		(0xAC01 <= code && code <= 0xAC1B) || // Lo  [27] HANGUL SYLLABLE GAG..HANGUL SYLLABLE GAH
		(0xAC1D <= code && code <= 0xAC37) || // Lo  [27] HANGUL SYLLABLE GAEG..HANGUL SYLLABLE GAEH
		(0xAC39 <= code && code <= 0xAC53) || // Lo  [27] HANGUL SYLLABLE GYAG..HANGUL SYLLABLE GYAH
		(0xAC55 <= code && code <= 0xAC6F) || // Lo  [27] HANGUL SYLLABLE GYAEG..HANGUL SYLLABLE GYAEH
		(0xAC71 <= code && code <= 0xAC8B) || // Lo  [27] HANGUL SYLLABLE GEOG..HANGUL SYLLABLE GEOH
		(0xAC8D <= code && code <= 0xACA7) || // Lo  [27] HANGUL SYLLABLE GEG..HANGUL SYLLABLE GEH
		(0xACA9 <= code && code <= 0xACC3) || // Lo  [27] HANGUL SYLLABLE GYEOG..HANGUL SYLLABLE GYEOH
		(0xACC5 <= code && code <= 0xACDF) || // Lo  [27] HANGUL SYLLABLE GYEG..HANGUL SYLLABLE GYEH
		(0xACE1 <= code && code <= 0xACFB) || // Lo  [27] HANGUL SYLLABLE GOG..HANGUL SYLLABLE GOH
		(0xACFD <= code && code <= 0xAD17) || // Lo  [27] HANGUL SYLLABLE GWAG..HANGUL SYLLABLE GWAH
		(0xAD19 <= code && code <= 0xAD33) || // Lo  [27] HANGUL SYLLABLE GWAEG..HANGUL SYLLABLE GWAEH
		(0xAD35 <= code && code <= 0xAD4F) || // Lo  [27] HANGUL SYLLABLE GOEG..HANGUL SYLLABLE GOEH
		(0xAD51 <= code && code <= 0xAD6B) || // Lo  [27] HANGUL SYLLABLE GYOG..HANGUL SYLLABLE GYOH
		(0xAD6D <= code && code <= 0xAD87) || // Lo  [27] HANGUL SYLLABLE GUG..HANGUL SYLLABLE GUH
		(0xAD89 <= code && code <= 0xADA3) || // Lo  [27] HANGUL SYLLABLE GWEOG..HANGUL SYLLABLE GWEOH
		(0xADA5 <= code && code <= 0xADBF) || // Lo  [27] HANGUL SYLLABLE GWEG..HANGUL SYLLABLE GWEH
		(0xADC1 <= code && code <= 0xADDB) || // Lo  [27] HANGUL SYLLABLE GWIG..HANGUL SYLLABLE GWIH
		(0xADDD <= code && code <= 0xADF7) || // Lo  [27] HANGUL SYLLABLE GYUG..HANGUL SYLLABLE GYUH
		(0xADF9 <= code && code <= 0xAE13) || // Lo  [27] HANGUL SYLLABLE GEUG..HANGUL SYLLABLE GEUH
		(0xAE15 <= code && code <= 0xAE2F) || // Lo  [27] HANGUL SYLLABLE GYIG..HANGUL SYLLABLE GYIH
		(0xAE31 <= code && code <= 0xAE4B) || // Lo  [27] HANGUL SYLLABLE GIG..HANGUL SYLLABLE GIH
		(0xAE4D <= code && code <= 0xAE67) || // Lo  [27] HANGUL SYLLABLE GGAG..HANGUL SYLLABLE GGAH
		(0xAE69 <= code && code <= 0xAE83) || // Lo  [27] HANGUL SYLLABLE GGAEG..HANGUL SYLLABLE GGAEH
		(0xAE85 <= code && code <= 0xAE9F) || // Lo  [27] HANGUL SYLLABLE GGYAG..HANGUL SYLLABLE GGYAH
		(0xAEA1 <= code && code <= 0xAEBB) || // Lo  [27] HANGUL SYLLABLE GGYAEG..HANGUL SYLLABLE GGYAEH
		(0xAEBD <= code && code <= 0xAED7) || // Lo  [27] HANGUL SYLLABLE GGEOG..HANGUL SYLLABLE GGEOH
		(0xAED9 <= code && code <= 0xAEF3) || // Lo  [27] HANGUL SYLLABLE GGEG..HANGUL SYLLABLE GGEH
		(0xAEF5 <= code && code <= 0xAF0F) || // Lo  [27] HANGUL SYLLABLE GGYEOG..HANGUL SYLLABLE GGYEOH
		(0xAF11 <= code && code <= 0xAF2B) || // Lo  [27] HANGUL SYLLABLE GGYEG..HANGUL SYLLABLE GGYEH
		(0xAF2D <= code && code <= 0xAF47) || // Lo  [27] HANGUL SYLLABLE GGOG..HANGUL SYLLABLE GGOH
		(0xAF49 <= code && code <= 0xAF63) || // Lo  [27] HANGUL SYLLABLE GGWAG..HANGUL SYLLABLE GGWAH
		(0xAF65 <= code && code <= 0xAF7F) || // Lo  [27] HANGUL SYLLABLE GGWAEG..HANGUL SYLLABLE GGWAEH
		(0xAF81 <= code && code <= 0xAF9B) || // Lo  [27] HANGUL SYLLABLE GGOEG..HANGUL SYLLABLE GGOEH
		(0xAF9D <= code && code <= 0xAFB7) || // Lo  [27] HANGUL SYLLABLE GGYOG..HANGUL SYLLABLE GGYOH
		(0xAFB9 <= code && code <= 0xAFD3) || // Lo  [27] HANGUL SYLLABLE GGUG..HANGUL SYLLABLE GGUH
		(0xAFD5 <= code && code <= 0xAFEF) || // Lo  [27] HANGUL SYLLABLE GGWEOG..HANGUL SYLLABLE GGWEOH
		(0xAFF1 <= code && code <= 0xB00B) || // Lo  [27] HANGUL SYLLABLE GGWEG..HANGUL SYLLABLE GGWEH
		(0xB00D <= code && code <= 0xB027) || // Lo  [27] HANGUL SYLLABLE GGWIG..HANGUL SYLLABLE GGWIH
		(0xB029 <= code && code <= 0xB043) || // Lo  [27] HANGUL SYLLABLE GGYUG..HANGUL SYLLABLE GGYUH
		(0xB045 <= code && code <= 0xB05F) || // Lo  [27] HANGUL SYLLABLE GGEUG..HANGUL SYLLABLE GGEUH
		(0xB061 <= code && code <= 0xB07B) || // Lo  [27] HANGUL SYLLABLE GGYIG..HANGUL SYLLABLE GGYIH
		(0xB07D <= code && code <= 0xB097) || // Lo  [27] HANGUL SYLLABLE GGIG..HANGUL SYLLABLE GGIH
		(0xB099 <= code && code <= 0xB0B3) || // Lo  [27] HANGUL SYLLABLE NAG..HANGUL SYLLABLE NAH
		(0xB0B5 <= code && code <= 0xB0CF) || // Lo  [27] HANGUL SYLLABLE NAEG..HANGUL SYLLABLE NAEH
		(0xB0D1 <= code && code <= 0xB0EB) || // Lo  [27] HANGUL SYLLABLE NYAG..HANGUL SYLLABLE NYAH
		(0xB0ED <= code && code <= 0xB107) || // Lo  [27] HANGUL SYLLABLE NYAEG..HANGUL SYLLABLE NYAEH
		(0xB109 <= code && code <= 0xB123) || // Lo  [27] HANGUL SYLLABLE NEOG..HANGUL SYLLABLE NEOH
		(0xB125 <= code && code <= 0xB13F) || // Lo  [27] HANGUL SYLLABLE NEG..HANGUL SYLLABLE NEH
		(0xB141 <= code && code <= 0xB15B) || // Lo  [27] HANGUL SYLLABLE NYEOG..HANGUL SYLLABLE NYEOH
		(0xB15D <= code && code <= 0xB177) || // Lo  [27] HANGUL SYLLABLE NYEG..HANGUL SYLLABLE NYEH
		(0xB179 <= code && code <= 0xB193) || // Lo  [27] HANGUL SYLLABLE NOG..HANGUL SYLLABLE NOH
		(0xB195 <= code && code <= 0xB1AF) || // Lo  [27] HANGUL SYLLABLE NWAG..HANGUL SYLLABLE NWAH
		(0xB1B1 <= code && code <= 0xB1CB) || // Lo  [27] HANGUL SYLLABLE NWAEG..HANGUL SYLLABLE NWAEH
		(0xB1CD <= code && code <= 0xB1E7) || // Lo  [27] HANGUL SYLLABLE NOEG..HANGUL SYLLABLE NOEH
		(0xB1E9 <= code && code <= 0xB203) || // Lo  [27] HANGUL SYLLABLE NYOG..HANGUL SYLLABLE NYOH
		(0xB205 <= code && code <= 0xB21F) || // Lo  [27] HANGUL SYLLABLE NUG..HANGUL SYLLABLE NUH
		(0xB221 <= code && code <= 0xB23B) || // Lo  [27] HANGUL SYLLABLE NWEOG..HANGUL SYLLABLE NWEOH
		(0xB23D <= code && code <= 0xB257) || // Lo  [27] HANGUL SYLLABLE NWEG..HANGUL SYLLABLE NWEH
		(0xB259 <= code && code <= 0xB273) || // Lo  [27] HANGUL SYLLABLE NWIG..HANGUL SYLLABLE NWIH
		(0xB275 <= code && code <= 0xB28F) || // Lo  [27] HANGUL SYLLABLE NYUG..HANGUL SYLLABLE NYUH
		(0xB291 <= code && code <= 0xB2AB) || // Lo  [27] HANGUL SYLLABLE NEUG..HANGUL SYLLABLE NEUH
		(0xB2AD <= code && code <= 0xB2C7) || // Lo  [27] HANGUL SYLLABLE NYIG..HANGUL SYLLABLE NYIH
		(0xB2C9 <= code && code <= 0xB2E3) || // Lo  [27] HANGUL SYLLABLE NIG..HANGUL SYLLABLE NIH
		(0xB2E5 <= code && code <= 0xB2FF) || // Lo  [27] HANGUL SYLLABLE DAG..HANGUL SYLLABLE DAH
		(0xB301 <= code && code <= 0xB31B) || // Lo  [27] HANGUL SYLLABLE DAEG..HANGUL SYLLABLE DAEH
		(0xB31D <= code && code <= 0xB337) || // Lo  [27] HANGUL SYLLABLE DYAG..HANGUL SYLLABLE DYAH
		(0xB339 <= code && code <= 0xB353) || // Lo  [27] HANGUL SYLLABLE DYAEG..HANGUL SYLLABLE DYAEH
		(0xB355 <= code && code <= 0xB36F) || // Lo  [27] HANGUL SYLLABLE DEOG..HANGUL SYLLABLE DEOH
		(0xB371 <= code && code <= 0xB38B) || // Lo  [27] HANGUL SYLLABLE DEG..HANGUL SYLLABLE DEH
		(0xB38D <= code && code <= 0xB3A7) || // Lo  [27] HANGUL SYLLABLE DYEOG..HANGUL SYLLABLE DYEOH
		(0xB3A9 <= code && code <= 0xB3C3) || // Lo  [27] HANGUL SYLLABLE DYEG..HANGUL SYLLABLE DYEH
		(0xB3C5 <= code && code <= 0xB3DF) || // Lo  [27] HANGUL SYLLABLE DOG..HANGUL SYLLABLE DOH
		(0xB3E1 <= code && code <= 0xB3FB) || // Lo  [27] HANGUL SYLLABLE DWAG..HANGUL SYLLABLE DWAH
		(0xB3FD <= code && code <= 0xB417) || // Lo  [27] HANGUL SYLLABLE DWAEG..HANGUL SYLLABLE DWAEH
		(0xB419 <= code && code <= 0xB433) || // Lo  [27] HANGUL SYLLABLE DOEG..HANGUL SYLLABLE DOEH
		(0xB435 <= code && code <= 0xB44F) || // Lo  [27] HANGUL SYLLABLE DYOG..HANGUL SYLLABLE DYOH
		(0xB451 <= code && code <= 0xB46B) || // Lo  [27] HANGUL SYLLABLE DUG..HANGUL SYLLABLE DUH
		(0xB46D <= code && code <= 0xB487) || // Lo  [27] HANGUL SYLLABLE DWEOG..HANGUL SYLLABLE DWEOH
		(0xB489 <= code && code <= 0xB4A3) || // Lo  [27] HANGUL SYLLABLE DWEG..HANGUL SYLLABLE DWEH
		(0xB4A5 <= code && code <= 0xB4BF) || // Lo  [27] HANGUL SYLLABLE DWIG..HANGUL SYLLABLE DWIH
		(0xB4C1 <= code && code <= 0xB4DB) || // Lo  [27] HANGUL SYLLABLE DYUG..HANGUL SYLLABLE DYUH
		(0xB4DD <= code && code <= 0xB4F7) || // Lo  [27] HANGUL SYLLABLE DEUG..HANGUL SYLLABLE DEUH
		(0xB4F9 <= code && code <= 0xB513) || // Lo  [27] HANGUL SYLLABLE DYIG..HANGUL SYLLABLE DYIH
		(0xB515 <= code && code <= 0xB52F) || // Lo  [27] HANGUL SYLLABLE DIG..HANGUL SYLLABLE DIH
		(0xB531 <= code && code <= 0xB54B) || // Lo  [27] HANGUL SYLLABLE DDAG..HANGUL SYLLABLE DDAH
		(0xB54D <= code && code <= 0xB567) || // Lo  [27] HANGUL SYLLABLE DDAEG..HANGUL SYLLABLE DDAEH
		(0xB569 <= code && code <= 0xB583) || // Lo  [27] HANGUL SYLLABLE DDYAG..HANGUL SYLLABLE DDYAH
		(0xB585 <= code && code <= 0xB59F) || // Lo  [27] HANGUL SYLLABLE DDYAEG..HANGUL SYLLABLE DDYAEH
		(0xB5A1 <= code && code <= 0xB5BB) || // Lo  [27] HANGUL SYLLABLE DDEOG..HANGUL SYLLABLE DDEOH
		(0xB5BD <= code && code <= 0xB5D7) || // Lo  [27] HANGUL SYLLABLE DDEG..HANGUL SYLLABLE DDEH
		(0xB5D9 <= code && code <= 0xB5F3) || // Lo  [27] HANGUL SYLLABLE DDYEOG..HANGUL SYLLABLE DDYEOH
		(0xB5F5 <= code && code <= 0xB60F) || // Lo  [27] HANGUL SYLLABLE DDYEG..HANGUL SYLLABLE DDYEH
		(0xB611 <= code && code <= 0xB62B) || // Lo  [27] HANGUL SYLLABLE DDOG..HANGUL SYLLABLE DDOH
		(0xB62D <= code && code <= 0xB647) || // Lo  [27] HANGUL SYLLABLE DDWAG..HANGUL SYLLABLE DDWAH
		(0xB649 <= code && code <= 0xB663) || // Lo  [27] HANGUL SYLLABLE DDWAEG..HANGUL SYLLABLE DDWAEH
		(0xB665 <= code && code <= 0xB67F) || // Lo  [27] HANGUL SYLLABLE DDOEG..HANGUL SYLLABLE DDOEH
		(0xB681 <= code && code <= 0xB69B) || // Lo  [27] HANGUL SYLLABLE DDYOG..HANGUL SYLLABLE DDYOH
		(0xB69D <= code && code <= 0xB6B7) || // Lo  [27] HANGUL SYLLABLE DDUG..HANGUL SYLLABLE DDUH
		(0xB6B9 <= code && code <= 0xB6D3) || // Lo  [27] HANGUL SYLLABLE DDWEOG..HANGUL SYLLABLE DDWEOH
		(0xB6D5 <= code && code <= 0xB6EF) || // Lo  [27] HANGUL SYLLABLE DDWEG..HANGUL SYLLABLE DDWEH
		(0xB6F1 <= code && code <= 0xB70B) || // Lo  [27] HANGUL SYLLABLE DDWIG..HANGUL SYLLABLE DDWIH
		(0xB70D <= code && code <= 0xB727) || // Lo  [27] HANGUL SYLLABLE DDYUG..HANGUL SYLLABLE DDYUH
		(0xB729 <= code && code <= 0xB743) || // Lo  [27] HANGUL SYLLABLE DDEUG..HANGUL SYLLABLE DDEUH
		(0xB745 <= code && code <= 0xB75F) || // Lo  [27] HANGUL SYLLABLE DDYIG..HANGUL SYLLABLE DDYIH
		(0xB761 <= code && code <= 0xB77B) || // Lo  [27] HANGUL SYLLABLE DDIG..HANGUL SYLLABLE DDIH
		(0xB77D <= code && code <= 0xB797) || // Lo  [27] HANGUL SYLLABLE RAG..HANGUL SYLLABLE RAH
		(0xB799 <= code && code <= 0xB7B3) || // Lo  [27] HANGUL SYLLABLE RAEG..HANGUL SYLLABLE RAEH
		(0xB7B5 <= code && code <= 0xB7CF) || // Lo  [27] HANGUL SYLLABLE RYAG..HANGUL SYLLABLE RYAH
		(0xB7D1 <= code && code <= 0xB7EB) || // Lo  [27] HANGUL SYLLABLE RYAEG..HANGUL SYLLABLE RYAEH
		(0xB7ED <= code && code <= 0xB807) || // Lo  [27] HANGUL SYLLABLE REOG..HANGUL SYLLABLE REOH
		(0xB809 <= code && code <= 0xB823) || // Lo  [27] HANGUL SYLLABLE REG..HANGUL SYLLABLE REH
		(0xB825 <= code && code <= 0xB83F) || // Lo  [27] HANGUL SYLLABLE RYEOG..HANGUL SYLLABLE RYEOH
		(0xB841 <= code && code <= 0xB85B) || // Lo  [27] HANGUL SYLLABLE RYEG..HANGUL SYLLABLE RYEH
		(0xB85D <= code && code <= 0xB877) || // Lo  [27] HANGUL SYLLABLE ROG..HANGUL SYLLABLE ROH
		(0xB879 <= code && code <= 0xB893) || // Lo  [27] HANGUL SYLLABLE RWAG..HANGUL SYLLABLE RWAH
		(0xB895 <= code && code <= 0xB8AF) || // Lo  [27] HANGUL SYLLABLE RWAEG..HANGUL SYLLABLE RWAEH
		(0xB8B1 <= code && code <= 0xB8CB) || // Lo  [27] HANGUL SYLLABLE ROEG..HANGUL SYLLABLE ROEH
		(0xB8CD <= code && code <= 0xB8E7) || // Lo  [27] HANGUL SYLLABLE RYOG..HANGUL SYLLABLE RYOH
		(0xB8E9 <= code && code <= 0xB903) || // Lo  [27] HANGUL SYLLABLE RUG..HANGUL SYLLABLE RUH
		(0xB905 <= code && code <= 0xB91F) || // Lo  [27] HANGUL SYLLABLE RWEOG..HANGUL SYLLABLE RWEOH
		(0xB921 <= code && code <= 0xB93B) || // Lo  [27] HANGUL SYLLABLE RWEG..HANGUL SYLLABLE RWEH
		(0xB93D <= code && code <= 0xB957) || // Lo  [27] HANGUL SYLLABLE RWIG..HANGUL SYLLABLE RWIH
		(0xB959 <= code && code <= 0xB973) || // Lo  [27] HANGUL SYLLABLE RYUG..HANGUL SYLLABLE RYUH
		(0xB975 <= code && code <= 0xB98F) || // Lo  [27] HANGUL SYLLABLE REUG..HANGUL SYLLABLE REUH
		(0xB991 <= code && code <= 0xB9AB) || // Lo  [27] HANGUL SYLLABLE RYIG..HANGUL SYLLABLE RYIH
		(0xB9AD <= code && code <= 0xB9C7) || // Lo  [27] HANGUL SYLLABLE RIG..HANGUL SYLLABLE RIH
		(0xB9C9 <= code && code <= 0xB9E3) || // Lo  [27] HANGUL SYLLABLE MAG..HANGUL SYLLABLE MAH
		(0xB9E5 <= code && code <= 0xB9FF) || // Lo  [27] HANGUL SYLLABLE MAEG..HANGUL SYLLABLE MAEH
		(0xBA01 <= code && code <= 0xBA1B) || // Lo  [27] HANGUL SYLLABLE MYAG..HANGUL SYLLABLE MYAH
		(0xBA1D <= code && code <= 0xBA37) || // Lo  [27] HANGUL SYLLABLE MYAEG..HANGUL SYLLABLE MYAEH
		(0xBA39 <= code && code <= 0xBA53) || // Lo  [27] HANGUL SYLLABLE MEOG..HANGUL SYLLABLE MEOH
		(0xBA55 <= code && code <= 0xBA6F) || // Lo  [27] HANGUL SYLLABLE MEG..HANGUL SYLLABLE MEH
		(0xBA71 <= code && code <= 0xBA8B) || // Lo  [27] HANGUL SYLLABLE MYEOG..HANGUL SYLLABLE MYEOH
		(0xBA8D <= code && code <= 0xBAA7) || // Lo  [27] HANGUL SYLLABLE MYEG..HANGUL SYLLABLE MYEH
		(0xBAA9 <= code && code <= 0xBAC3) || // Lo  [27] HANGUL SYLLABLE MOG..HANGUL SYLLABLE MOH
		(0xBAC5 <= code && code <= 0xBADF) || // Lo  [27] HANGUL SYLLABLE MWAG..HANGUL SYLLABLE MWAH
		(0xBAE1 <= code && code <= 0xBAFB) || // Lo  [27] HANGUL SYLLABLE MWAEG..HANGUL SYLLABLE MWAEH
		(0xBAFD <= code && code <= 0xBB17) || // Lo  [27] HANGUL SYLLABLE MOEG..HANGUL SYLLABLE MOEH
		(0xBB19 <= code && code <= 0xBB33) || // Lo  [27] HANGUL SYLLABLE MYOG..HANGUL SYLLABLE MYOH
		(0xBB35 <= code && code <= 0xBB4F) || // Lo  [27] HANGUL SYLLABLE MUG..HANGUL SYLLABLE MUH
		(0xBB51 <= code && code <= 0xBB6B) || // Lo  [27] HANGUL SYLLABLE MWEOG..HANGUL SYLLABLE MWEOH
		(0xBB6D <= code && code <= 0xBB87) || // Lo  [27] HANGUL SYLLABLE MWEG..HANGUL SYLLABLE MWEH
		(0xBB89 <= code && code <= 0xBBA3) || // Lo  [27] HANGUL SYLLABLE MWIG..HANGUL SYLLABLE MWIH
		(0xBBA5 <= code && code <= 0xBBBF) || // Lo  [27] HANGUL SYLLABLE MYUG..HANGUL SYLLABLE MYUH
		(0xBBC1 <= code && code <= 0xBBDB) || // Lo  [27] HANGUL SYLLABLE MEUG..HANGUL SYLLABLE MEUH
		(0xBBDD <= code && code <= 0xBBF7) || // Lo  [27] HANGUL SYLLABLE MYIG..HANGUL SYLLABLE MYIH
		(0xBBF9 <= code && code <= 0xBC13) || // Lo  [27] HANGUL SYLLABLE MIG..HANGUL SYLLABLE MIH
		(0xBC15 <= code && code <= 0xBC2F) || // Lo  [27] HANGUL SYLLABLE BAG..HANGUL SYLLABLE BAH
		(0xBC31 <= code && code <= 0xBC4B) || // Lo  [27] HANGUL SYLLABLE BAEG..HANGUL SYLLABLE BAEH
		(0xBC4D <= code && code <= 0xBC67) || // Lo  [27] HANGUL SYLLABLE BYAG..HANGUL SYLLABLE BYAH
		(0xBC69 <= code && code <= 0xBC83) || // Lo  [27] HANGUL SYLLABLE BYAEG..HANGUL SYLLABLE BYAEH
		(0xBC85 <= code && code <= 0xBC9F) || // Lo  [27] HANGUL SYLLABLE BEOG..HANGUL SYLLABLE BEOH
		(0xBCA1 <= code && code <= 0xBCBB) || // Lo  [27] HANGUL SYLLABLE BEG..HANGUL SYLLABLE BEH
		(0xBCBD <= code && code <= 0xBCD7) || // Lo  [27] HANGUL SYLLABLE BYEOG..HANGUL SYLLABLE BYEOH
		(0xBCD9 <= code && code <= 0xBCF3) || // Lo  [27] HANGUL SYLLABLE BYEG..HANGUL SYLLABLE BYEH
		(0xBCF5 <= code && code <= 0xBD0F) || // Lo  [27] HANGUL SYLLABLE BOG..HANGUL SYLLABLE BOH
		(0xBD11 <= code && code <= 0xBD2B) || // Lo  [27] HANGUL SYLLABLE BWAG..HANGUL SYLLABLE BWAH
		(0xBD2D <= code && code <= 0xBD47) || // Lo  [27] HANGUL SYLLABLE BWAEG..HANGUL SYLLABLE BWAEH
		(0xBD49 <= code && code <= 0xBD63) || // Lo  [27] HANGUL SYLLABLE BOEG..HANGUL SYLLABLE BOEH
		(0xBD65 <= code && code <= 0xBD7F) || // Lo  [27] HANGUL SYLLABLE BYOG..HANGUL SYLLABLE BYOH
		(0xBD81 <= code && code <= 0xBD9B) || // Lo  [27] HANGUL SYLLABLE BUG..HANGUL SYLLABLE BUH
		(0xBD9D <= code && code <= 0xBDB7) || // Lo  [27] HANGUL SYLLABLE BWEOG..HANGUL SYLLABLE BWEOH
		(0xBDB9 <= code && code <= 0xBDD3) || // Lo  [27] HANGUL SYLLABLE BWEG..HANGUL SYLLABLE BWEH
		(0xBDD5 <= code && code <= 0xBDEF) || // Lo  [27] HANGUL SYLLABLE BWIG..HANGUL SYLLABLE BWIH
		(0xBDF1 <= code && code <= 0xBE0B) || // Lo  [27] HANGUL SYLLABLE BYUG..HANGUL SYLLABLE BYUH
		(0xBE0D <= code && code <= 0xBE27) || // Lo  [27] HANGUL SYLLABLE BEUG..HANGUL SYLLABLE BEUH
		(0xBE29 <= code && code <= 0xBE43) || // Lo  [27] HANGUL SYLLABLE BYIG..HANGUL SYLLABLE BYIH
		(0xBE45 <= code && code <= 0xBE5F) || // Lo  [27] HANGUL SYLLABLE BIG..HANGUL SYLLABLE BIH
		(0xBE61 <= code && code <= 0xBE7B) || // Lo  [27] HANGUL SYLLABLE BBAG..HANGUL SYLLABLE BBAH
		(0xBE7D <= code && code <= 0xBE97) || // Lo  [27] HANGUL SYLLABLE BBAEG..HANGUL SYLLABLE BBAEH
		(0xBE99 <= code && code <= 0xBEB3) || // Lo  [27] HANGUL SYLLABLE BBYAG..HANGUL SYLLABLE BBYAH
		(0xBEB5 <= code && code <= 0xBECF) || // Lo  [27] HANGUL SYLLABLE BBYAEG..HANGUL SYLLABLE BBYAEH
		(0xBED1 <= code && code <= 0xBEEB) || // Lo  [27] HANGUL SYLLABLE BBEOG..HANGUL SYLLABLE BBEOH
		(0xBEED <= code && code <= 0xBF07) || // Lo  [27] HANGUL SYLLABLE BBEG..HANGUL SYLLABLE BBEH
		(0xBF09 <= code && code <= 0xBF23) || // Lo  [27] HANGUL SYLLABLE BBYEOG..HANGUL SYLLABLE BBYEOH
		(0xBF25 <= code && code <= 0xBF3F) || // Lo  [27] HANGUL SYLLABLE BBYEG..HANGUL SYLLABLE BBYEH
		(0xBF41 <= code && code <= 0xBF5B) || // Lo  [27] HANGUL SYLLABLE BBOG..HANGUL SYLLABLE BBOH
		(0xBF5D <= code && code <= 0xBF77) || // Lo  [27] HANGUL SYLLABLE BBWAG..HANGUL SYLLABLE BBWAH
		(0xBF79 <= code && code <= 0xBF93) || // Lo  [27] HANGUL SYLLABLE BBWAEG..HANGUL SYLLABLE BBWAEH
		(0xBF95 <= code && code <= 0xBFAF) || // Lo  [27] HANGUL SYLLABLE BBOEG..HANGUL SYLLABLE BBOEH
		(0xBFB1 <= code && code <= 0xBFCB) || // Lo  [27] HANGUL SYLLABLE BBYOG..HANGUL SYLLABLE BBYOH
		(0xBFCD <= code && code <= 0xBFE7) || // Lo  [27] HANGUL SYLLABLE BBUG..HANGUL SYLLABLE BBUH
		(0xBFE9 <= code && code <= 0xC003) || // Lo  [27] HANGUL SYLLABLE BBWEOG..HANGUL SYLLABLE BBWEOH
		(0xC005 <= code && code <= 0xC01F) || // Lo  [27] HANGUL SYLLABLE BBWEG..HANGUL SYLLABLE BBWEH
		(0xC021 <= code && code <= 0xC03B) || // Lo  [27] HANGUL SYLLABLE BBWIG..HANGUL SYLLABLE BBWIH
		(0xC03D <= code && code <= 0xC057) || // Lo  [27] HANGUL SYLLABLE BBYUG..HANGUL SYLLABLE BBYUH
		(0xC059 <= code && code <= 0xC073) || // Lo  [27] HANGUL SYLLABLE BBEUG..HANGUL SYLLABLE BBEUH
		(0xC075 <= code && code <= 0xC08F) || // Lo  [27] HANGUL SYLLABLE BBYIG..HANGUL SYLLABLE BBYIH
		(0xC091 <= code && code <= 0xC0AB) || // Lo  [27] HANGUL SYLLABLE BBIG..HANGUL SYLLABLE BBIH
		(0xC0AD <= code && code <= 0xC0C7) || // Lo  [27] HANGUL SYLLABLE SAG..HANGUL SYLLABLE SAH
		(0xC0C9 <= code && code <= 0xC0E3) || // Lo  [27] HANGUL SYLLABLE SAEG..HANGUL SYLLABLE SAEH
		(0xC0E5 <= code && code <= 0xC0FF) || // Lo  [27] HANGUL SYLLABLE SYAG..HANGUL SYLLABLE SYAH
		(0xC101 <= code && code <= 0xC11B) || // Lo  [27] HANGUL SYLLABLE SYAEG..HANGUL SYLLABLE SYAEH
		(0xC11D <= code && code <= 0xC137) || // Lo  [27] HANGUL SYLLABLE SEOG..HANGUL SYLLABLE SEOH
		(0xC139 <= code && code <= 0xC153) || // Lo  [27] HANGUL SYLLABLE SEG..HANGUL SYLLABLE SEH
		(0xC155 <= code && code <= 0xC16F) || // Lo  [27] HANGUL SYLLABLE SYEOG..HANGUL SYLLABLE SYEOH
		(0xC171 <= code && code <= 0xC18B) || // Lo  [27] HANGUL SYLLABLE SYEG..HANGUL SYLLABLE SYEH
		(0xC18D <= code && code <= 0xC1A7) || // Lo  [27] HANGUL SYLLABLE SOG..HANGUL SYLLABLE SOH
		(0xC1A9 <= code && code <= 0xC1C3) || // Lo  [27] HANGUL SYLLABLE SWAG..HANGUL SYLLABLE SWAH
		(0xC1C5 <= code && code <= 0xC1DF) || // Lo  [27] HANGUL SYLLABLE SWAEG..HANGUL SYLLABLE SWAEH
		(0xC1E1 <= code && code <= 0xC1FB) || // Lo  [27] HANGUL SYLLABLE SOEG..HANGUL SYLLABLE SOEH
		(0xC1FD <= code && code <= 0xC217) || // Lo  [27] HANGUL SYLLABLE SYOG..HANGUL SYLLABLE SYOH
		(0xC219 <= code && code <= 0xC233) || // Lo  [27] HANGUL SYLLABLE SUG..HANGUL SYLLABLE SUH
		(0xC235 <= code && code <= 0xC24F) || // Lo  [27] HANGUL SYLLABLE SWEOG..HANGUL SYLLABLE SWEOH
		(0xC251 <= code && code <= 0xC26B) || // Lo  [27] HANGUL SYLLABLE SWEG..HANGUL SYLLABLE SWEH
		(0xC26D <= code && code <= 0xC287) || // Lo  [27] HANGUL SYLLABLE SWIG..HANGUL SYLLABLE SWIH
		(0xC289 <= code && code <= 0xC2A3) || // Lo  [27] HANGUL SYLLABLE SYUG..HANGUL SYLLABLE SYUH
		(0xC2A5 <= code && code <= 0xC2BF) || // Lo  [27] HANGUL SYLLABLE SEUG..HANGUL SYLLABLE SEUH
		(0xC2C1 <= code && code <= 0xC2DB) || // Lo  [27] HANGUL SYLLABLE SYIG..HANGUL SYLLABLE SYIH
		(0xC2DD <= code && code <= 0xC2F7) || // Lo  [27] HANGUL SYLLABLE SIG..HANGUL SYLLABLE SIH
		(0xC2F9 <= code && code <= 0xC313) || // Lo  [27] HANGUL SYLLABLE SSAG..HANGUL SYLLABLE SSAH
		(0xC315 <= code && code <= 0xC32F) || // Lo  [27] HANGUL SYLLABLE SSAEG..HANGUL SYLLABLE SSAEH
		(0xC331 <= code && code <= 0xC34B) || // Lo  [27] HANGUL SYLLABLE SSYAG..HANGUL SYLLABLE SSYAH
		(0xC34D <= code && code <= 0xC367) || // Lo  [27] HANGUL SYLLABLE SSYAEG..HANGUL SYLLABLE SSYAEH
		(0xC369 <= code && code <= 0xC383) || // Lo  [27] HANGUL SYLLABLE SSEOG..HANGUL SYLLABLE SSEOH
		(0xC385 <= code && code <= 0xC39F) || // Lo  [27] HANGUL SYLLABLE SSEG..HANGUL SYLLABLE SSEH
		(0xC3A1 <= code && code <= 0xC3BB) || // Lo  [27] HANGUL SYLLABLE SSYEOG..HANGUL SYLLABLE SSYEOH
		(0xC3BD <= code && code <= 0xC3D7) || // Lo  [27] HANGUL SYLLABLE SSYEG..HANGUL SYLLABLE SSYEH
		(0xC3D9 <= code && code <= 0xC3F3) || // Lo  [27] HANGUL SYLLABLE SSOG..HANGUL SYLLABLE SSOH
		(0xC3F5 <= code && code <= 0xC40F) || // Lo  [27] HANGUL SYLLABLE SSWAG..HANGUL SYLLABLE SSWAH
		(0xC411 <= code && code <= 0xC42B) || // Lo  [27] HANGUL SYLLABLE SSWAEG..HANGUL SYLLABLE SSWAEH
		(0xC42D <= code && code <= 0xC447) || // Lo  [27] HANGUL SYLLABLE SSOEG..HANGUL SYLLABLE SSOEH
		(0xC449 <= code && code <= 0xC463) || // Lo  [27] HANGUL SYLLABLE SSYOG..HANGUL SYLLABLE SSYOH
		(0xC465 <= code && code <= 0xC47F) || // Lo  [27] HANGUL SYLLABLE SSUG..HANGUL SYLLABLE SSUH
		(0xC481 <= code && code <= 0xC49B) || // Lo  [27] HANGUL SYLLABLE SSWEOG..HANGUL SYLLABLE SSWEOH
		(0xC49D <= code && code <= 0xC4B7) || // Lo  [27] HANGUL SYLLABLE SSWEG..HANGUL SYLLABLE SSWEH
		(0xC4B9 <= code && code <= 0xC4D3) || // Lo  [27] HANGUL SYLLABLE SSWIG..HANGUL SYLLABLE SSWIH
		(0xC4D5 <= code && code <= 0xC4EF) || // Lo  [27] HANGUL SYLLABLE SSYUG..HANGUL SYLLABLE SSYUH
		(0xC4F1 <= code && code <= 0xC50B) || // Lo  [27] HANGUL SYLLABLE SSEUG..HANGUL SYLLABLE SSEUH
		(0xC50D <= code && code <= 0xC527) || // Lo  [27] HANGUL SYLLABLE SSYIG..HANGUL SYLLABLE SSYIH
		(0xC529 <= code && code <= 0xC543) || // Lo  [27] HANGUL SYLLABLE SSIG..HANGUL SYLLABLE SSIH
		(0xC545 <= code && code <= 0xC55F) || // Lo  [27] HANGUL SYLLABLE AG..HANGUL SYLLABLE AH
		(0xC561 <= code && code <= 0xC57B) || // Lo  [27] HANGUL SYLLABLE AEG..HANGUL SYLLABLE AEH
		(0xC57D <= code && code <= 0xC597) || // Lo  [27] HANGUL SYLLABLE YAG..HANGUL SYLLABLE YAH
		(0xC599 <= code && code <= 0xC5B3) || // Lo  [27] HANGUL SYLLABLE YAEG..HANGUL SYLLABLE YAEH
		(0xC5B5 <= code && code <= 0xC5CF) || // Lo  [27] HANGUL SYLLABLE EOG..HANGUL SYLLABLE EOH
		(0xC5D1 <= code && code <= 0xC5EB) || // Lo  [27] HANGUL SYLLABLE EG..HANGUL SYLLABLE EH
		(0xC5ED <= code && code <= 0xC607) || // Lo  [27] HANGUL SYLLABLE YEOG..HANGUL SYLLABLE YEOH
		(0xC609 <= code && code <= 0xC623) || // Lo  [27] HANGUL SYLLABLE YEG..HANGUL SYLLABLE YEH
		(0xC625 <= code && code <= 0xC63F) || // Lo  [27] HANGUL SYLLABLE OG..HANGUL SYLLABLE OH
		(0xC641 <= code && code <= 0xC65B) || // Lo  [27] HANGUL SYLLABLE WAG..HANGUL SYLLABLE WAH
		(0xC65D <= code && code <= 0xC677) || // Lo  [27] HANGUL SYLLABLE WAEG..HANGUL SYLLABLE WAEH
		(0xC679 <= code && code <= 0xC693) || // Lo  [27] HANGUL SYLLABLE OEG..HANGUL SYLLABLE OEH
		(0xC695 <= code && code <= 0xC6AF) || // Lo  [27] HANGUL SYLLABLE YOG..HANGUL SYLLABLE YOH
		(0xC6B1 <= code && code <= 0xC6CB) || // Lo  [27] HANGUL SYLLABLE UG..HANGUL SYLLABLE UH
		(0xC6CD <= code && code <= 0xC6E7) || // Lo  [27] HANGUL SYLLABLE WEOG..HANGUL SYLLABLE WEOH
		(0xC6E9 <= code && code <= 0xC703) || // Lo  [27] HANGUL SYLLABLE WEG..HANGUL SYLLABLE WEH
		(0xC705 <= code && code <= 0xC71F) || // Lo  [27] HANGUL SYLLABLE WIG..HANGUL SYLLABLE WIH
		(0xC721 <= code && code <= 0xC73B) || // Lo  [27] HANGUL SYLLABLE YUG..HANGUL SYLLABLE YUH
		(0xC73D <= code && code <= 0xC757) || // Lo  [27] HANGUL SYLLABLE EUG..HANGUL SYLLABLE EUH
		(0xC759 <= code && code <= 0xC773) || // Lo  [27] HANGUL SYLLABLE YIG..HANGUL SYLLABLE YIH
		(0xC775 <= code && code <= 0xC78F) || // Lo  [27] HANGUL SYLLABLE IG..HANGUL SYLLABLE IH
		(0xC791 <= code && code <= 0xC7AB) || // Lo  [27] HANGUL SYLLABLE JAG..HANGUL SYLLABLE JAH
		(0xC7AD <= code && code <= 0xC7C7) || // Lo  [27] HANGUL SYLLABLE JAEG..HANGUL SYLLABLE JAEH
		(0xC7C9 <= code && code <= 0xC7E3) || // Lo  [27] HANGUL SYLLABLE JYAG..HANGUL SYLLABLE JYAH
		(0xC7E5 <= code && code <= 0xC7FF) || // Lo  [27] HANGUL SYLLABLE JYAEG..HANGUL SYLLABLE JYAEH
		(0xC801 <= code && code <= 0xC81B) || // Lo  [27] HANGUL SYLLABLE JEOG..HANGUL SYLLABLE JEOH
		(0xC81D <= code && code <= 0xC837) || // Lo  [27] HANGUL SYLLABLE JEG..HANGUL SYLLABLE JEH
		(0xC839 <= code && code <= 0xC853) || // Lo  [27] HANGUL SYLLABLE JYEOG..HANGUL SYLLABLE JYEOH
		(0xC855 <= code && code <= 0xC86F) || // Lo  [27] HANGUL SYLLABLE JYEG..HANGUL SYLLABLE JYEH
		(0xC871 <= code && code <= 0xC88B) || // Lo  [27] HANGUL SYLLABLE JOG..HANGUL SYLLABLE JOH
		(0xC88D <= code && code <= 0xC8A7) || // Lo  [27] HANGUL SYLLABLE JWAG..HANGUL SYLLABLE JWAH
		(0xC8A9 <= code && code <= 0xC8C3) || // Lo  [27] HANGUL SYLLABLE JWAEG..HANGUL SYLLABLE JWAEH
		(0xC8C5 <= code && code <= 0xC8DF) || // Lo  [27] HANGUL SYLLABLE JOEG..HANGUL SYLLABLE JOEH
		(0xC8E1 <= code && code <= 0xC8FB) || // Lo  [27] HANGUL SYLLABLE JYOG..HANGUL SYLLABLE JYOH
		(0xC8FD <= code && code <= 0xC917) || // Lo  [27] HANGUL SYLLABLE JUG..HANGUL SYLLABLE JUH
		(0xC919 <= code && code <= 0xC933) || // Lo  [27] HANGUL SYLLABLE JWEOG..HANGUL SYLLABLE JWEOH
		(0xC935 <= code && code <= 0xC94F) || // Lo  [27] HANGUL SYLLABLE JWEG..HANGUL SYLLABLE JWEH
		(0xC951 <= code && code <= 0xC96B) || // Lo  [27] HANGUL SYLLABLE JWIG..HANGUL SYLLABLE JWIH
		(0xC96D <= code && code <= 0xC987) || // Lo  [27] HANGUL SYLLABLE JYUG..HANGUL SYLLABLE JYUH
		(0xC989 <= code && code <= 0xC9A3) || // Lo  [27] HANGUL SYLLABLE JEUG..HANGUL SYLLABLE JEUH
		(0xC9A5 <= code && code <= 0xC9BF) || // Lo  [27] HANGUL SYLLABLE JYIG..HANGUL SYLLABLE JYIH
		(0xC9C1 <= code && code <= 0xC9DB) || // Lo  [27] HANGUL SYLLABLE JIG..HANGUL SYLLABLE JIH
		(0xC9DD <= code && code <= 0xC9F7) || // Lo  [27] HANGUL SYLLABLE JJAG..HANGUL SYLLABLE JJAH
		(0xC9F9 <= code && code <= 0xCA13) || // Lo  [27] HANGUL SYLLABLE JJAEG..HANGUL SYLLABLE JJAEH
		(0xCA15 <= code && code <= 0xCA2F) || // Lo  [27] HANGUL SYLLABLE JJYAG..HANGUL SYLLABLE JJYAH
		(0xCA31 <= code && code <= 0xCA4B) || // Lo  [27] HANGUL SYLLABLE JJYAEG..HANGUL SYLLABLE JJYAEH
		(0xCA4D <= code && code <= 0xCA67) || // Lo  [27] HANGUL SYLLABLE JJEOG..HANGUL SYLLABLE JJEOH
		(0xCA69 <= code && code <= 0xCA83) || // Lo  [27] HANGUL SYLLABLE JJEG..HANGUL SYLLABLE JJEH
		(0xCA85 <= code && code <= 0xCA9F) || // Lo  [27] HANGUL SYLLABLE JJYEOG..HANGUL SYLLABLE JJYEOH
		(0xCAA1 <= code && code <= 0xCABB) || // Lo  [27] HANGUL SYLLABLE JJYEG..HANGUL SYLLABLE JJYEH
		(0xCABD <= code && code <= 0xCAD7) || // Lo  [27] HANGUL SYLLABLE JJOG..HANGUL SYLLABLE JJOH
		(0xCAD9 <= code && code <= 0xCAF3) || // Lo  [27] HANGUL SYLLABLE JJWAG..HANGUL SYLLABLE JJWAH
		(0xCAF5 <= code && code <= 0xCB0F) || // Lo  [27] HANGUL SYLLABLE JJWAEG..HANGUL SYLLABLE JJWAEH
		(0xCB11 <= code && code <= 0xCB2B) || // Lo  [27] HANGUL SYLLABLE JJOEG..HANGUL SYLLABLE JJOEH
		(0xCB2D <= code && code <= 0xCB47) || // Lo  [27] HANGUL SYLLABLE JJYOG..HANGUL SYLLABLE JJYOH
		(0xCB49 <= code && code <= 0xCB63) || // Lo  [27] HANGUL SYLLABLE JJUG..HANGUL SYLLABLE JJUH
		(0xCB65 <= code && code <= 0xCB7F) || // Lo  [27] HANGUL SYLLABLE JJWEOG..HANGUL SYLLABLE JJWEOH
		(0xCB81 <= code && code <= 0xCB9B) || // Lo  [27] HANGUL SYLLABLE JJWEG..HANGUL SYLLABLE JJWEH
		(0xCB9D <= code && code <= 0xCBB7) || // Lo  [27] HANGUL SYLLABLE JJWIG..HANGUL SYLLABLE JJWIH
		(0xCBB9 <= code && code <= 0xCBD3) || // Lo  [27] HANGUL SYLLABLE JJYUG..HANGUL SYLLABLE JJYUH
		(0xCBD5 <= code && code <= 0xCBEF) || // Lo  [27] HANGUL SYLLABLE JJEUG..HANGUL SYLLABLE JJEUH
		(0xCBF1 <= code && code <= 0xCC0B) || // Lo  [27] HANGUL SYLLABLE JJYIG..HANGUL SYLLABLE JJYIH
		(0xCC0D <= code && code <= 0xCC27) || // Lo  [27] HANGUL SYLLABLE JJIG..HANGUL SYLLABLE JJIH
		(0xCC29 <= code && code <= 0xCC43) || // Lo  [27] HANGUL SYLLABLE CAG..HANGUL SYLLABLE CAH
		(0xCC45 <= code && code <= 0xCC5F) || // Lo  [27] HANGUL SYLLABLE CAEG..HANGUL SYLLABLE CAEH
		(0xCC61 <= code && code <= 0xCC7B) || // Lo  [27] HANGUL SYLLABLE CYAG..HANGUL SYLLABLE CYAH
		(0xCC7D <= code && code <= 0xCC97) || // Lo  [27] HANGUL SYLLABLE CYAEG..HANGUL SYLLABLE CYAEH
		(0xCC99 <= code && code <= 0xCCB3) || // Lo  [27] HANGUL SYLLABLE CEOG..HANGUL SYLLABLE CEOH
		(0xCCB5 <= code && code <= 0xCCCF) || // Lo  [27] HANGUL SYLLABLE CEG..HANGUL SYLLABLE CEH
		(0xCCD1 <= code && code <= 0xCCEB) || // Lo  [27] HANGUL SYLLABLE CYEOG..HANGUL SYLLABLE CYEOH
		(0xCCED <= code && code <= 0xCD07) || // Lo  [27] HANGUL SYLLABLE CYEG..HANGUL SYLLABLE CYEH
		(0xCD09 <= code && code <= 0xCD23) || // Lo  [27] HANGUL SYLLABLE COG..HANGUL SYLLABLE COH
		(0xCD25 <= code && code <= 0xCD3F) || // Lo  [27] HANGUL SYLLABLE CWAG..HANGUL SYLLABLE CWAH
		(0xCD41 <= code && code <= 0xCD5B) || // Lo  [27] HANGUL SYLLABLE CWAEG..HANGUL SYLLABLE CWAEH
		(0xCD5D <= code && code <= 0xCD77) || // Lo  [27] HANGUL SYLLABLE COEG..HANGUL SYLLABLE COEH
		(0xCD79 <= code && code <= 0xCD93) || // Lo  [27] HANGUL SYLLABLE CYOG..HANGUL SYLLABLE CYOH
		(0xCD95 <= code && code <= 0xCDAF) || // Lo  [27] HANGUL SYLLABLE CUG..HANGUL SYLLABLE CUH
		(0xCDB1 <= code && code <= 0xCDCB) || // Lo  [27] HANGUL SYLLABLE CWEOG..HANGUL SYLLABLE CWEOH
		(0xCDCD <= code && code <= 0xCDE7) || // Lo  [27] HANGUL SYLLABLE CWEG..HANGUL SYLLABLE CWEH
		(0xCDE9 <= code && code <= 0xCE03) || // Lo  [27] HANGUL SYLLABLE CWIG..HANGUL SYLLABLE CWIH
		(0xCE05 <= code && code <= 0xCE1F) || // Lo  [27] HANGUL SYLLABLE CYUG..HANGUL SYLLABLE CYUH
		(0xCE21 <= code && code <= 0xCE3B) || // Lo  [27] HANGUL SYLLABLE CEUG..HANGUL SYLLABLE CEUH
		(0xCE3D <= code && code <= 0xCE57) || // Lo  [27] HANGUL SYLLABLE CYIG..HANGUL SYLLABLE CYIH
		(0xCE59 <= code && code <= 0xCE73) || // Lo  [27] HANGUL SYLLABLE CIG..HANGUL SYLLABLE CIH
		(0xCE75 <= code && code <= 0xCE8F) || // Lo  [27] HANGUL SYLLABLE KAG..HANGUL SYLLABLE KAH
		(0xCE91 <= code && code <= 0xCEAB) || // Lo  [27] HANGUL SYLLABLE KAEG..HANGUL SYLLABLE KAEH
		(0xCEAD <= code && code <= 0xCEC7) || // Lo  [27] HANGUL SYLLABLE KYAG..HANGUL SYLLABLE KYAH
		(0xCEC9 <= code && code <= 0xCEE3) || // Lo  [27] HANGUL SYLLABLE KYAEG..HANGUL SYLLABLE KYAEH
		(0xCEE5 <= code && code <= 0xCEFF) || // Lo  [27] HANGUL SYLLABLE KEOG..HANGUL SYLLABLE KEOH
		(0xCF01 <= code && code <= 0xCF1B) || // Lo  [27] HANGUL SYLLABLE KEG..HANGUL SYLLABLE KEH
		(0xCF1D <= code && code <= 0xCF37) || // Lo  [27] HANGUL SYLLABLE KYEOG..HANGUL SYLLABLE KYEOH
		(0xCF39 <= code && code <= 0xCF53) || // Lo  [27] HANGUL SYLLABLE KYEG..HANGUL SYLLABLE KYEH
		(0xCF55 <= code && code <= 0xCF6F) || // Lo  [27] HANGUL SYLLABLE KOG..HANGUL SYLLABLE KOH
		(0xCF71 <= code && code <= 0xCF8B) || // Lo  [27] HANGUL SYLLABLE KWAG..HANGUL SYLLABLE KWAH
		(0xCF8D <= code && code <= 0xCFA7) || // Lo  [27] HANGUL SYLLABLE KWAEG..HANGUL SYLLABLE KWAEH
		(0xCFA9 <= code && code <= 0xCFC3) || // Lo  [27] HANGUL SYLLABLE KOEG..HANGUL SYLLABLE KOEH
		(0xCFC5 <= code && code <= 0xCFDF) || // Lo  [27] HANGUL SYLLABLE KYOG..HANGUL SYLLABLE KYOH
		(0xCFE1 <= code && code <= 0xCFFB) || // Lo  [27] HANGUL SYLLABLE KUG..HANGUL SYLLABLE KUH
		(0xCFFD <= code && code <= 0xD017) || // Lo  [27] HANGUL SYLLABLE KWEOG..HANGUL SYLLABLE KWEOH
		(0xD019 <= code && code <= 0xD033) || // Lo  [27] HANGUL SYLLABLE KWEG..HANGUL SYLLABLE KWEH
		(0xD035 <= code && code <= 0xD04F) || // Lo  [27] HANGUL SYLLABLE KWIG..HANGUL SYLLABLE KWIH
		(0xD051 <= code && code <= 0xD06B) || // Lo  [27] HANGUL SYLLABLE KYUG..HANGUL SYLLABLE KYUH
		(0xD06D <= code && code <= 0xD087) || // Lo  [27] HANGUL SYLLABLE KEUG..HANGUL SYLLABLE KEUH
		(0xD089 <= code && code <= 0xD0A3) || // Lo  [27] HANGUL SYLLABLE KYIG..HANGUL SYLLABLE KYIH
		(0xD0A5 <= code && code <= 0xD0BF) || // Lo  [27] HANGUL SYLLABLE KIG..HANGUL SYLLABLE KIH
		(0xD0C1 <= code && code <= 0xD0DB) || // Lo  [27] HANGUL SYLLABLE TAG..HANGUL SYLLABLE TAH
		(0xD0DD <= code && code <= 0xD0F7) || // Lo  [27] HANGUL SYLLABLE TAEG..HANGUL SYLLABLE TAEH
		(0xD0F9 <= code && code <= 0xD113) || // Lo  [27] HANGUL SYLLABLE TYAG..HANGUL SYLLABLE TYAH
		(0xD115 <= code && code <= 0xD12F) || // Lo  [27] HANGUL SYLLABLE TYAEG..HANGUL SYLLABLE TYAEH
		(0xD131 <= code && code <= 0xD14B) || // Lo  [27] HANGUL SYLLABLE TEOG..HANGUL SYLLABLE TEOH
		(0xD14D <= code && code <= 0xD167) || // Lo  [27] HANGUL SYLLABLE TEG..HANGUL SYLLABLE TEH
		(0xD169 <= code && code <= 0xD183) || // Lo  [27] HANGUL SYLLABLE TYEOG..HANGUL SYLLABLE TYEOH
		(0xD185 <= code && code <= 0xD19F) || // Lo  [27] HANGUL SYLLABLE TYEG..HANGUL SYLLABLE TYEH
		(0xD1A1 <= code && code <= 0xD1BB) || // Lo  [27] HANGUL SYLLABLE TOG..HANGUL SYLLABLE TOH
		(0xD1BD <= code && code <= 0xD1D7) || // Lo  [27] HANGUL SYLLABLE TWAG..HANGUL SYLLABLE TWAH
		(0xD1D9 <= code && code <= 0xD1F3) || // Lo  [27] HANGUL SYLLABLE TWAEG..HANGUL SYLLABLE TWAEH
		(0xD1F5 <= code && code <= 0xD20F) || // Lo  [27] HANGUL SYLLABLE TOEG..HANGUL SYLLABLE TOEH
		(0xD211 <= code && code <= 0xD22B) || // Lo  [27] HANGUL SYLLABLE TYOG..HANGUL SYLLABLE TYOH
		(0xD22D <= code && code <= 0xD247) || // Lo  [27] HANGUL SYLLABLE TUG..HANGUL SYLLABLE TUH
		(0xD249 <= code && code <= 0xD263) || // Lo  [27] HANGUL SYLLABLE TWEOG..HANGUL SYLLABLE TWEOH
		(0xD265 <= code && code <= 0xD27F) || // Lo  [27] HANGUL SYLLABLE TWEG..HANGUL SYLLABLE TWEH
		(0xD281 <= code && code <= 0xD29B) || // Lo  [27] HANGUL SYLLABLE TWIG..HANGUL SYLLABLE TWIH
		(0xD29D <= code && code <= 0xD2B7) || // Lo  [27] HANGUL SYLLABLE TYUG..HANGUL SYLLABLE TYUH
		(0xD2B9 <= code && code <= 0xD2D3) || // Lo  [27] HANGUL SYLLABLE TEUG..HANGUL SYLLABLE TEUH
		(0xD2D5 <= code && code <= 0xD2EF) || // Lo  [27] HANGUL SYLLABLE TYIG..HANGUL SYLLABLE TYIH
		(0xD2F1 <= code && code <= 0xD30B) || // Lo  [27] HANGUL SYLLABLE TIG..HANGUL SYLLABLE TIH
		(0xD30D <= code && code <= 0xD327) || // Lo  [27] HANGUL SYLLABLE PAG..HANGUL SYLLABLE PAH
		(0xD329 <= code && code <= 0xD343) || // Lo  [27] HANGUL SYLLABLE PAEG..HANGUL SYLLABLE PAEH
		(0xD345 <= code && code <= 0xD35F) || // Lo  [27] HANGUL SYLLABLE PYAG..HANGUL SYLLABLE PYAH
		(0xD361 <= code && code <= 0xD37B) || // Lo  [27] HANGUL SYLLABLE PYAEG..HANGUL SYLLABLE PYAEH
		(0xD37D <= code && code <= 0xD397) || // Lo  [27] HANGUL SYLLABLE PEOG..HANGUL SYLLABLE PEOH
		(0xD399 <= code && code <= 0xD3B3) || // Lo  [27] HANGUL SYLLABLE PEG..HANGUL SYLLABLE PEH
		(0xD3B5 <= code && code <= 0xD3CF) || // Lo  [27] HANGUL SYLLABLE PYEOG..HANGUL SYLLABLE PYEOH
		(0xD3D1 <= code && code <= 0xD3EB) || // Lo  [27] HANGUL SYLLABLE PYEG..HANGUL SYLLABLE PYEH
		(0xD3ED <= code && code <= 0xD407) || // Lo  [27] HANGUL SYLLABLE POG..HANGUL SYLLABLE POH
		(0xD409 <= code && code <= 0xD423) || // Lo  [27] HANGUL SYLLABLE PWAG..HANGUL SYLLABLE PWAH
		(0xD425 <= code && code <= 0xD43F) || // Lo  [27] HANGUL SYLLABLE PWAEG..HANGUL SYLLABLE PWAEH
		(0xD441 <= code && code <= 0xD45B) || // Lo  [27] HANGUL SYLLABLE POEG..HANGUL SYLLABLE POEH
		(0xD45D <= code && code <= 0xD477) || // Lo  [27] HANGUL SYLLABLE PYOG..HANGUL SYLLABLE PYOH
		(0xD479 <= code && code <= 0xD493) || // Lo  [27] HANGUL SYLLABLE PUG..HANGUL SYLLABLE PUH
		(0xD495 <= code && code <= 0xD4AF) || // Lo  [27] HANGUL SYLLABLE PWEOG..HANGUL SYLLABLE PWEOH
		(0xD4B1 <= code && code <= 0xD4CB) || // Lo  [27] HANGUL SYLLABLE PWEG..HANGUL SYLLABLE PWEH
		(0xD4CD <= code && code <= 0xD4E7) || // Lo  [27] HANGUL SYLLABLE PWIG..HANGUL SYLLABLE PWIH
		(0xD4E9 <= code && code <= 0xD503) || // Lo  [27] HANGUL SYLLABLE PYUG..HANGUL SYLLABLE PYUH
		(0xD505 <= code && code <= 0xD51F) || // Lo  [27] HANGUL SYLLABLE PEUG..HANGUL SYLLABLE PEUH
		(0xD521 <= code && code <= 0xD53B) || // Lo  [27] HANGUL SYLLABLE PYIG..HANGUL SYLLABLE PYIH
		(0xD53D <= code && code <= 0xD557) || // Lo  [27] HANGUL SYLLABLE PIG..HANGUL SYLLABLE PIH
		(0xD559 <= code && code <= 0xD573) || // Lo  [27] HANGUL SYLLABLE HAG..HANGUL SYLLABLE HAH
		(0xD575 <= code && code <= 0xD58F) || // Lo  [27] HANGUL SYLLABLE HAEG..HANGUL SYLLABLE HAEH
		(0xD591 <= code && code <= 0xD5AB) || // Lo  [27] HANGUL SYLLABLE HYAG..HANGUL SYLLABLE HYAH
		(0xD5AD <= code && code <= 0xD5C7) || // Lo  [27] HANGUL SYLLABLE HYAEG..HANGUL SYLLABLE HYAEH
		(0xD5C9 <= code && code <= 0xD5E3) || // Lo  [27] HANGUL SYLLABLE HEOG..HANGUL SYLLABLE HEOH
		(0xD5E5 <= code && code <= 0xD5FF) || // Lo  [27] HANGUL SYLLABLE HEG..HANGUL SYLLABLE HEH
		(0xD601 <= code && code <= 0xD61B) || // Lo  [27] HANGUL SYLLABLE HYEOG..HANGUL SYLLABLE HYEOH
		(0xD61D <= code && code <= 0xD637) || // Lo  [27] HANGUL SYLLABLE HYEG..HANGUL SYLLABLE HYEH
		(0xD639 <= code && code <= 0xD653) || // Lo  [27] HANGUL SYLLABLE HOG..HANGUL SYLLABLE HOH
		(0xD655 <= code && code <= 0xD66F) || // Lo  [27] HANGUL SYLLABLE HWAG..HANGUL SYLLABLE HWAH
		(0xD671 <= code && code <= 0xD68B) || // Lo  [27] HANGUL SYLLABLE HWAEG..HANGUL SYLLABLE HWAEH
		(0xD68D <= code && code <= 0xD6A7) || // Lo  [27] HANGUL SYLLABLE HOEG..HANGUL SYLLABLE HOEH
		(0xD6A9 <= code && code <= 0xD6C3) || // Lo  [27] HANGUL SYLLABLE HYOG..HANGUL SYLLABLE HYOH
		(0xD6C5 <= code && code <= 0xD6DF) || // Lo  [27] HANGUL SYLLABLE HUG..HANGUL SYLLABLE HUH
		(0xD6E1 <= code && code <= 0xD6FB) || // Lo  [27] HANGUL SYLLABLE HWEOG..HANGUL SYLLABLE HWEOH
		(0xD6FD <= code && code <= 0xD717) || // Lo  [27] HANGUL SYLLABLE HWEG..HANGUL SYLLABLE HWEH
		(0xD719 <= code && code <= 0xD733) || // Lo  [27] HANGUL SYLLABLE HWIG..HANGUL SYLLABLE HWIH
		(0xD735 <= code && code <= 0xD74F) || // Lo  [27] HANGUL SYLLABLE HYUG..HANGUL SYLLABLE HYUH
		(0xD751 <= code && code <= 0xD76B) || // Lo  [27] HANGUL SYLLABLE HEUG..HANGUL SYLLABLE HEUH
		(0xD76D <= code && code <= 0xD787) || // Lo  [27] HANGUL SYLLABLE HYIG..HANGUL SYLLABLE HYIH
		(0xD789 <= code && code <= 0xD7A3) // Lo  [27] HANGUL SYLLABLE HIG..HANGUL SYLLABLE HIH
		){
			return LVT;
		}
		
		if(
		0x261D == code || // So       WHITE UP POINTING INDEX
		0x26F9 == code || // So       PERSON WITH BALL
		(0x270A <= code && code <= 0x270D) || // So   [4] RAISED FIST..WRITING HAND
		0x1F385 == code || // So       FATHER CHRISTMAS
		(0x1F3C2 <= code && code <= 0x1F3C4) || // So   [3] SNOWBOARDER..SURFER
		0x1F3C7 == code || // So       HORSE RACING
		(0x1F3CA <= code && code <= 0x1F3CC) || // So   [3] SWIMMER..GOLFER
		(0x1F442 <= code && code <= 0x1F443) || // So   [2] EAR..NOSE
		(0x1F446 <= code && code <= 0x1F450) || // So  [11] WHITE UP POINTING BACKHAND INDEX..OPEN HANDS SIGN
		0x1F46E == code || // So       POLICE OFFICER
		(0x1F470 <= code && code <= 0x1F478) || // So   [9] BRIDE WITH VEIL..PRINCESS
		0x1F47C == code || // So       BABY ANGEL
		(0x1F481 <= code && code <= 0x1F483) || // So   [3] INFORMATION DESK PERSON..DANCER
		(0x1F485 <= code && code <= 0x1F487) || // So   [3] NAIL POLISH..HAIRCUT
		0x1F4AA == code || // So       FLEXED BICEPS
		(0x1F574 <= code && code <= 0x1F575) || // So   [2] MAN IN BUSINESS SUIT LEVITATING..SLEUTH OR SPY
		0x1F57A == code || // So       MAN DANCING
		0x1F590 == code || // So       RAISED HAND WITH FINGERS SPLAYED
		(0x1F595 <= code && code <= 0x1F596) || // So   [2] REVERSED HAND WITH MIDDLE FINGER EXTENDED..RAISED HAND WITH PART BETWEEN MIDDLE AND RING FINGERS
		(0x1F645 <= code && code <= 0x1F647) || // So   [3] FACE WITH NO GOOD GESTURE..PERSON BOWING DEEPLY
		(0x1F64B <= code && code <= 0x1F64F) || // So   [5] HAPPY PERSON RAISING ONE HAND..PERSON WITH FOLDED HANDS
		0x1F6A3 == code || // So       ROWBOAT
		(0x1F6B4 <= code && code <= 0x1F6B6) || // So   [3] BICYCLIST..PEDESTRIAN
		0x1F6C0 == code || // So       BATH
		0x1F6CC == code || // So       SLEEPING ACCOMMODATION
		(0x1F918 <= code && code <= 0x1F91C) || // So   [5] SIGN OF THE HORNS..RIGHT-FACING FIST
		(0x1F91E <= code && code <= 0x1F91F) || // So   [2] HAND WITH INDEX AND MIDDLE FINGERS CROSSED..I LOVE YOU HAND SIGN
		0x1F926 == code || // So       FACE PALM
		(0x1F930 <= code && code <= 0x1F939) || // So  [10] PREGNANT WOMAN..JUGGLING
		(0x1F93D <= code && code <= 0x1F93E) || // So   [2] WATER POLO..HANDBALL
		(0x1F9D1 <= code && code <= 0x1F9DD) // So  [13] ADULT..ELF
		){
			return E_Base;
		}

		if(
		(0x1F3FB <= code && code <= 0x1F3FF) // Sk   [5] EMOJI MODIFIER FITZPATRICK TYPE-1-2..EMOJI MODIFIER FITZPATRICK TYPE-6
		){
			return E_Modifier;
		}

		if(
		0x200D == code // Cf       ZERO WIDTH JOINER
		){
			return ZWJ;
		}

		if(
		0x2640 == code || // So       FEMALE SIGN
		0x2642 == code || // So       MALE SIGN
		(0x2695 <= code && code <= 0x2696) || // So   [2] STAFF OF AESCULAPIUS..SCALES
		0x2708 == code || // So       AIRPLANE
		0x2764 == code || // So       HEAVY BLACK HEART
		0x1F308 == code || // So       RAINBOW
		0x1F33E == code || // So       EAR OF RICE
		0x1F373 == code || // So       COOKING
		0x1F393 == code || // So       GRADUATION CAP
		0x1F3A4 == code || // So       MICROPHONE
		0x1F3A8 == code || // So       ARTIST PALETTE
		0x1F3EB == code || // So       SCHOOL
		0x1F3ED == code || // So       FACTORY
		0x1F48B == code || // So       KISS MARK
		(0x1F4BB <= code && code <= 0x1F4BC) || // So   [2] PERSONAL COMPUTER..BRIEFCASE
		0x1F527 == code || // So       WRENCH
		0x1F52C == code || // So       MICROSCOPE
		0x1F5E8 == code || // So       LEFT SPEECH BUBBLE
		0x1F680 == code || // So       ROCKET
		0x1F692 == code // So       FIRE ENGINE
		){
			return Glue_After_Zwj;
		}

		if(
		(0x1F466 <= code && code <= 0x1F469) // So   [4] BOY..WOMAN
		){
			return E_Base_GAZ;
		}
		
		
		//all unlisted characters have a grapheme break property of "Other"
		return Other;
	}
	return this;
}

if (typeof module != 'undefined' && module.exports) {
    module.exports = GraphemeSplitter;
}
