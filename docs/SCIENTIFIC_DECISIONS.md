# Scientific decisions and current limitations

This file records decisions that materially affect the present circuit. The aim is to make it possible for another researcher to understand what was observed in MICrONS, what was changed for technical reasons, and what remains an assumption.

## 1. Structural source

The current circuit uses MICrONS CAVE as the structural source and pins the analysis to a specific materialisation and skeleton version through configuration.

Reason: MICrONS segmentation and annotations evolve with proofreading. A versioned analysis is necessary for reproducibility.

## 2. Functional mapping is used for identity, not yet for activity fitting

Manual coregistration is used to restrict the candidate population to neurons that have functional identifiers.

All mapping rows are retained. Two of the selected 20 neurons currently have more than one functional mapping.

The functional traces themselves are not part of the current processed dataset. Acquisition is planned from MICrONS NDA v8.

## 3. DANDI is not used as the current functional-selection gate

The published NWB files and the current CAVE manual-coregistration table do not provide complete one-to-one coverage under the current stable identifiers. The repository therefore does not discard structurally eligible neurons simply because the older embedded NWB coregistration does not resolve them.

DANDI remains useful for cross-version inspection. The planned current functional source is NDA v8 using the preserved `(session, scan_idx, unit_id)` identities.

## 4. Conservative axon proofreading strategy

The present selection uses the configured conservative axon strategy.

This is a policy choice, not a hard-coded property of the software. A user can broaden the allowed strategy in `configs/project.yaml` and rerun the pipeline.

Changing the axon strategy can change the candidate pool, recording ranking and final 20-neuron population, so it should be treated as a new analysis rather than a cosmetic setting.

## 5. Apical dendrite labels

The pipeline preserves SWC type 4 when supplied but does not require it.

None of the current 20 processed morphologies contains type-4 apical labels.

This should be documented as "no separate apical label in the source skeletons", not as proof that these neurons biologically lack apical dendrites.

## 6. Zero-radius samples

Four raw radius values were zero:

- one point in model 1;
- one point in model 11;
- two points in model 17.

All four are isolated terminal tips.

Only the radius was replaced, using positive topological boundary information. Coordinates, point connectivity and compartment type were unchanged. Raw source SWCs were not overwritten.

## 7. Model 11 dendrite-to-axon transition

Model 11 (nucleus 262893) contains one proximal type-3 to type-2 transition.

Additional graph inspection showed:

```text
axon entry points: 1
connected axon components: 1
```

The transition is therefore preserved as source information instead of being relabelled simply to satisfy an expected morphology pattern.

It remains appropriate to review the original reconstruction/annotation if this cell becomes important for high-fidelity axonal modelling.

## 8. Synapse placement

The 54 recurrent contacts are not collapsed into 38 graph edges for simulation.

Every anatomical contact remains separate and retains a saved postsynaptic section ID and section position.

Current target-compartment mapping contains 52 dendritic and 2 axonal recurrent contacts. The two axonal targets should be checked against the relevant MICrONS target-structure annotations before a later high-fidelity release.

## 9. "Outside selected 20" is not the same as "extrinsic"

There are 44,282 incoming contacts from 29,991 positive presynaptic roots outside the selected 20-cell population.

These include any neuron not selected into the explicit 20-cell circuit. A local V1 neuron can therefore be "outside selected 20" while still being local cortical input.

Future extrinsic-input work must classify these sources rather than treating all 29,991 roots as one biological category.

## 10. Global morphology coordinates

The processed SWCs preserve the MICrONS coordinate frame.

The structural SONATA uses `recenter=0`. Soma coordinates are retained as metadata and are not applied again as a translation.

## 11. Allen reference cell model

The current electrical reference is Allen neuronal model `487245719`, specimen `486893033`, a VISp Layer-4 `Scnn1a-Tg3-Cre` biophysical perisomatic model.

This parameter set was fitted to the Allen specimen, not to the selected MICrONS neurons.

It is used because the same-cell MICrONS data available to this project are calcium imaging rather than patch-clamp measurements.

## 12. Preserving the MICrONS axon during parameter transfer

BMTK's standard Allen perisomatic processing replaces the reconstructed axon with a short standardised axon representation.

That behaviour would remove morphology needed for the MICrONS structural circuit.

The project therefore uses a custom cell processor that applies the Allen-derived perisomatic parameters while leaving the CAVE-derived morphology intact.

This makes the resulting cells transferred reference models, not reproductions of the original Allen fit.

## 13. Independent current injection

A 0.60 nA current step is currently used to exercise the transferred MICrONS cell models.

This value was selected because the transferred model 0 remained subthreshold at 0.20 nA but fired at 0.60 nA.

It is not presented as a measured biological input for these MICrONS neurons.

## 14. Recurrent synaptic physiology

The first connected simulation uses a simple excitatory `Exp2Syn` model:

```text
tau1 = 0.2 ms
tau2 = 1.8 ms
Erev = 0 mV
weight = 0.001 µS
delay = 2.0 ms
```

These parameters are reference values used to test the connected-circuit implementation.

No claim is made that the individual MICrONS synapses have these exact conductances, kinetics or delays.

## 15. BMTK 1.2 compatibility

Two run-local adaptations are currently required:

- BMTK 1.2 reads an edge-index dataset name differently from the current structural SONATA representation;
- HDF5 morphology text can reach the Allen cell loader as bytes, so run-local morphology names are moved to a text node-types table.

Neither adaptation changes the structural SONATA package.

Before a stable release, this logic should be isolated in a clearly documented compatibility module and covered by tests.
