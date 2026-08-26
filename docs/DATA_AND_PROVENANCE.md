# Data and provenance

## MICrONS data

The structural pipeline uses MICrONS CAVE data with the materialisation and skeleton version specified in `configs/project.yaml`.

MICrONS data are versioned because proofreading and annotations continue to evolve. Reproducible analyses should therefore record the materialisation/version used rather than relying on "latest".

The current repository uses:

```text
CAVE materialisation: 1822
skeleton version:     4
```

## Functional identity

The selected-neuron manifest preserves MICrONS functional identifiers.

The processed functional mapping table is:

```text
data/processed/final20_functional_mappings.parquet
```

It contains all retained mapping rows rather than one artificially chosen unit per neuron.

The activity traces are not yet stored in this repository.

## Planned NDA v8 source

Current MICrONS functional data are available through the NDA/DataJoint database.

The planned retrieval path uses the saved session/scan/unit identifiers and NDA v8 tables such as `ScanUnit`, `Fluorescence` and `Activity`.

## MICrONS attribution

MICrONS Explorer states that its site material is available under Creative Commons Attribution 4.0 and provides a specific citation policy.

For the cortical mm³ data, use the MICrONS Consortium flagship publication specified by the MICrONS citation policy.

If MICrONS-derived morphologies or tables are redistributed in this repository, include clear attribution and the applicable CC BY 4.0 notice.

## Allen model attribution

The preliminary electrical model uses Allen Cell Types neuronal model `487245719`.

For a public repository:

- document the neuronal model and specimen IDs;
- link users to the Allen Institute source/instructions;
- cite the Allen Institute according to its citation policy;
- avoid committing the downloaded model files until redistribution conditions and repository size have been reviewed.

The project code licence and third-party data licences are separate questions. Choosing an MIT/BSD/Apache licence for project code does not relicense MICrONS or Allen data.
