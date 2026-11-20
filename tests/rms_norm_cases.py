"""Build fixture signatures in the same interface order as Cannery intake."""

from pathlib import Path

import yaml


def signatures():
    root = Path(__file__).resolve().parents[1] / "catalog/rms_norm/spec"
    spec = yaml.safe_load((root / "cannery.yaml").read_text())["spec"]
    interface = spec["interface"]
    recipe_file = spec["cases"]["recipe_file"]
    contract_file = spec["cases"]["tensor_contract_file"]
    recipes = yaml.safe_load((root / recipe_file).read_text())["cases"]
    contracts = yaml.safe_load((root / contract_file).read_text())["cases"]
    if not recipes:
        raise ValueError("RMSNorm spec must include at least one case")

    for recipe in recipes:
        contract = contracts[recipe["id"]]
        yield {
            "inputs": [
                {"name": name, "shape": recipe["inputs"][name]["shape"], "dtype": recipe["inputs"][name]["dtype"]}
                for name in interface["inputs"]
            ],
            "outputs": [
                {"name": name, **contract["outputs"][name], "alias": contract["aliases"].get(name)}
                for name in interface["outputs"]
            ],
            "scalars": [{"name": name, **value} for name, value in sorted(contract["scalars"].items())],
        }
