#!/usr/bin/env python3
import os
import argparse
import re

def parse_textgrid_tier(content, offset):
    """Extrait les intervalles d'une tier et applique le décalage temporel."""
    # Trouve les blocs d'intervalles xmin, xmax, text
    interval_pattern = re.compile(
        r'intervals\s*\[\s*\d+\s*\]\s*:\s*xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*("(.*?)")'
    )

    intervals = []
    for match in interval_pattern.finditer(content):
        xmin = float(match.group(1)) + offset
        xmax = float(match.group(2)) + offset
        text = match.group(3) # Conserve les guillemets originaux
        intervals.append((xmin, xmax, text))
    return intervals

def merge_textgrids(files, output_path):
    all_intervals = {}
    current_offset = 0.0

    for file_path in files:
        if not os.path.exists(file_path):
            print(f"⚠️ Fichier introuvable : {file_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Trouver la durée totale de ce sous-TextGrid (xmax global)
        xmax_match = re.search(r'^xmax\s*=\s*([\d.]+)', content, re.MULTILINE)
        if not xmax_match:
            continue
        file_duration = float(xmax_match.group(1))

        # Extraire le nom de la tier (souvent "words" ou "phones" avec MFA)
        tier_matches = re.finditer(r'item\s*\[\s*\d+\s*\]\s*:\s*class\s*=\s*"IntervalTier"\s*name\s*=\s*"(.*?)"', content)

        # Découper le contenu par tier si plusieurs existent
        tier_splits = re.split(r'item\s*\[\s*\d+\s*\]\s*:', content)[1:]

        for idx, tier_match in enumerate(tier_matches):
            tier_name = tier_match.group(1)
            if tier_name not in all_intervals:
                all_intervals[tier_name] = []

            tier_content = tier_splits[idx] if idx < len(tier_splits) else ""
            intervals = parse_textgrid_tier(tier_content, current_offset)
            all_intervals[tier_name].extend(intervals)

        # Le début du prochain fichier correspond à la fin de celui-ci
        current_offset += file_duration

    # Écriture du fichier fusionné au format standard Praat
    total_duration = current_offset
    num_tiers = len(all_intervals)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("File type = \"ooTextFile\"\n")
        f.write("Object class = \"TextGrid\"\n\n")
        f.write("xmin = 0\n")
        f.write(f"xmax = {total_duration:.6f}\n")
        f.write("tiers? <exists>\n")
        f.write(f"size = {num_tiers}\n")
        f.write("item []:\n")

        for t_idx, (tier_name, intervals) in enumerate(all_intervals.items(), 1):
            f.write(f"    item [{t_idx}]:\n")
            f.write("        class = \"IntervalTier\"\n")
            f.write(f"        name = \"{tier_name}\"\n")
            f.write("        xmin = 0\n")
            f.write(f"        xmax = {total_duration:.6f}\n")
            f.write(f"        intervals: size = {len(intervals)}\n")

            for i_idx, (xmin, xmax, text) in enumerate(intervals, 1):
                f.write(f"            intervals [{i_idx}]:\n")
                f.write(f"                xmin = {xmin:.6f}\n")
                f.write(f"                xmax = {xmax:.6f}\n")
                f.write(f"                text = {text}\n")

    print(f"✅ TextGrid fusionné créé avec succès : {output_path} (Durée totale : {total_duration:.2f}s)")

def main():
    parser = argparse.ArgumentParser(description="Fusionne les fichiers TextGrid d'un dossier en un seul.")
    parser.add_argument("-i", "--input-dir", required=True, help="Dossier contenant les TextGrid à fusionner")
    parser.add_argument("-o", "--output", required=True, help="Chemin du TextGrid final fusionné")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        parser.error(f"Dossier introuvable : {args.input_dir}")

    out_abs = os.path.abspath(args.output)
    input_files = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".textgrid")
        and os.path.abspath(os.path.join(args.input_dir, f)) != out_abs
    )

    if not input_files:
        parser.error(f"Aucun fichier .TextGrid dans {args.input_dir}")

    merge_textgrids(input_files, args.output)

if __name__ == "__main__":
    main()
