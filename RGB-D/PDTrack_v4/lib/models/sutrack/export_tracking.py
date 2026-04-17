import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export unique sequence names from a JSON subset file.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Output TXT file.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sequences = sorted({item["sequence"] for item in data if "sequence" in item})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sequences) + "\n", encoding="utf-8")

    print(f"unique_sequences: {len(sequences)}")
    print(f"saved_to: {output_path}")


if __name__ == "__main__":
    main()