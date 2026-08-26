"""Custom BioNet cell processors for MICrONS demonstration simulations.

These processors deliberately keep the CAVE-derived morphology intact.
They do not replace the reconstructed axon with the two-section Allen axon stub.
"""

from __future__ import annotations

from typing import Any

from bmtk.simulator.core.pyfunction_cache import add_cell_processor


SECTION_PREFIXES = {"soma", "axon", "dend", "apic"}


def _section_class(section: Any) -> str:
    """Return the Allen-style four-character NEURON section class."""

    name = section.name()
    parts = name.split(".")
    local_name = parts[1] if len(parts) > 1 else parts[0]
    return local_name[:4]


def microns_perisomatic_preserve_morphology(
    hobj: Any,
    cell: Any,
    dynamics_params: dict[str, Any] | None,
) -> Any:
    """Apply Allen perisomatic parameters without altering CAVE morphology.

    This reproduces BMTK's parameter-assignment step for an Allen perisomatic
    model while intentionally omitting ``fix_axon_peri()``.

    Scientific status
    -----------------
    This is an Allen-derived provisional parameter transfer. The parameters
    were fitted to the Allen reference specimen, not to the MICrONS neuron.
    """

    if dynamics_params is None:
        raise ValueError(
            "microns_perisomatic_preserve_morphology requires dynamics parameters."
        )

    passive_entries = dynamics_params.get("passive", [])
    condition_entries = dynamics_params.get("conditions", [])
    genome = dynamics_params.get("genome", [])

    if len(passive_entries) != 1 or len(condition_entries) != 1:
        raise ValueError(
            "Expected exactly one passive and one conditions entry "
            "in the Allen fit-parameter file."
        )

    passive = passive_entries[0]
    conditions = condition_entries[0]

    cm_by_section = {
        str(entry["section"]): float(entry["cm"])
        for entry in passive["cm"]
    }

    sections_by_class: dict[str, list[Any]] = {}

    for section in hobj.all:
        section_class = _section_class(section)

        if section_class not in SECTION_PREFIXES:
            raise ValueError(
                f"Unexpected NEURON section class {section_class!r} "
                f"for section {section.name()!r}."
            )

        if section_class not in cm_by_section:
            raise ValueError(
                f"No Allen membrane capacitance parameter exists for "
                f"section class {section_class!r}."
            )

        sections_by_class.setdefault(section_class, []).append(section)

        section.Ra = float(passive["ra"])
        section.cm = cm_by_section[section_class]
        section.insert("pas")

        for segment in section:
            segment.pas.e = float(passive["e_pas"])

    for parameter in genome:
        section_class = str(parameter["section"])
        mechanism = str(parameter.get("mechanism", ""))
        parameter_name = str(parameter["name"])
        parameter_value = float(parameter["value"])

        for section in sections_by_class.get(section_class, []):
            if mechanism:
                section.insert(mechanism)

            setattr(
                section,
                parameter_name,
                parameter_value,
            )

    for reversal in conditions.get("erev", []):
        section_class = str(reversal["section"])

        for section in sections_by_class.get(section_class, []):
            section.ena = float(reversal["ena"])
            section.ek = float(reversal["ek"])

    return hobj


add_cell_processor(
    microns_perisomatic_preserve_morphology,
    name="microns_perisomatic_preserve_morphology",
    overwrite=True,
)
