# Dataset access and provenance

This project uses third-party retinal fundus-image datasets. Raw images are not redistributed by this repository.

## APTOS 2019

Use the official/original distribution available to the research team. Record in the final public release:

- source URL or persistent landing page;
- date accessed;
- license / terms of use;
- original file count used;
- exclusions, if any;
- train/validation/test split procedure;
- class-label mapping.

Local directory:

```text
data/aptos2019/
```

## EyePACS

Use the official/original distribution available to the research team. Record:

- source URL or landing page;
- date accessed;
- terms of use;
- selected subset, if applicable;
- exclusions;
- split procedure;
- class-label mapping.

Local directory:

```text
data/eyepacs/
```

## Messidor-2

Messidor-2 is used as an independent external evaluation dataset. Record:

- source URL or landing page;
- date accessed;
- license / access terms;
- label source and mapping;
- any preprocessing needed before evaluation;
- exclusions, if any.

Local directory:

```text
data/messidor2/
```

## Important

Do not upload raw images or restricted metadata unless the corresponding dataset license explicitly permits redistribution.

For full reproducibility, release split manifests containing non-sensitive image identifiers and labels whenever the original dataset terms permit it.
