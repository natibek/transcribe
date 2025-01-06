#!/usr/bin/env bash
while read line; do
  if [[ "$line" == *"madmom"* ]]; then
    madmom_version="$line"
  else
    pip install $line
  fi
done < requirements.txt

pip install "$madmom_version"

