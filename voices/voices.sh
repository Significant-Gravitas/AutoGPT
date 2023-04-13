#!/bin/sh

# Liste der Namen
names="catherine martha maeve riya nora leona gordon arthur aidan akash aaron xander"

# Präfix für say-Befehl
prefix="com.apple.speech.synthesis.voice.custom.siri"

# Leeres Array für funktionierende Namen
declare -a working_names=()

# Teste jeden Namen und füge funktionierende Namen zum Array hinzu
for name in $names; do
  for variant in "" ".enhanced" ".premium"; do
    full_name="${prefix}.${name}${variant}"
    if say -v "$full_name" "hello" 2>/dev/null; then
      working_names+=("$full_name")
    fi
  done
done

# Schreibe funktionierende Namen in eine Datei
printf "%s\n" "${working_names[@]}" > working_names.txt

echo "Funktionierende Namen wurden in working_names.txt gespeichert."
