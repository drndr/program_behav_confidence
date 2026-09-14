"""Structural RE2 checks for inspection, not automatic semantic validation."""

import json
import re
import zipfile


def input_text(row):
    """The final INPUT block belongs to the task in this release's templates."""
    blocks = re.findall(r"\[INPUT\](.*?)\[/INPUT\]", row["messages"][0]["content"], re.S)
    if not blocks:
        raise ValueError(f"No input block: {row['id']}")
    return re.sub(r"^```(?:json|python)?\s*|\s*```$", "", blocks[-1].strip()).strip()


def inspect_re2(rows, cruxeval_ids, upstream_zip):
    results = []
    with zipfile.ZipFile(upstream_zip) as archive:
        for row in rows:
            prompt = row["messages"][0]["content"]
            addresses = re.findall(r" at (0x[0-9a-fA-F]+)", str(row["target_value"]))
            item = {
                "id": row["id"], "source": row["source_dataset"], "group": row["complexity_group"],
                "cruxeval_source": row["source_dataset"] == "cruxeval",
                "shared_cruxeval_id": row["source_dataset"] == "cruxeval" and row["id"] in cruxeval_ids,
                "unexposed_address": any(address not in prompt for address in addresses),
                "invalid_input_json": False, "double_encoded_stdin": False,
                "input_text": input_text(row), "parse_error": None,
            }
            try:
                parsed = json.loads(item["input_text"])
            except json.JSONDecodeError as error:
                item["invalid_input_json"] = True
                item["parse_error"] = str(error)
            else:
                if row["source_dataset"] == "Avatar":
                    path = (f"RE2-Bench-{row['artifact_commit']}/dataset/re2-bench/"
                            f"input-output/Avatar/{row['upstream_case_name']}")
                    original = json.loads(archive.read(path).decode().splitlines()[row["upstream_io_index"]])["values"]["input"]
                    item["double_encoded_stdin"] = parsed == {"input": json.dumps(original, indent=4)}
                    item["upstream_stdin"] = original
            results.append(item)
    return results
