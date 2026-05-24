# pheno-extract-ai

Software tools, synthetic clinical notes, gold-standard annotations, and evaluation scripts for a study of Human Phenotype Ontology (HPO) phenotype extraction using large language models with and without HPO MCP tooling.

## Data availability

All software tools, synthetic clinical notes, gold-standard annotations, and evaluation scripts used in this study are publicly available via GitHub at:

- https://github.com/clinical-mcp/pheno-extract-ai
- https://github.com/clinical-mcp/hpo_mcp

Raw model outputs and combined analysis data tables are archived separately in Dryad. See `dryad/README.md` and replace the DOI placeholder after Dryad publication.

## Repository structure

```text
extraction_scripts/          Model/API extraction scripts
synthetic_clinical_notes/    Synthetic note inputs used in the study
gold_standard_annotations/   Gold-standard/manual-adjudication CSVs by note
evaluation_scripts/          Validation, adjudication-support, and analysis scripts
dryad/                       Pointer to Dryad data archive
```

## Running the extraction scripts

The extraction scripts are configured to use standard provider environment variables:

- `ANTHROPIC_API_KEY` for Claude scripts
- `OPENAI_API_KEY` for GPT/OpenAI scripts
- `GEMINI_API_KEY` for Gemini scripts
- `XAI_API_KEY` for Grok/xAI scripts

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

- The clinical notes included here are synthetic study inputs. No protected health information is intended to be included.
- Large/raw result logs, cost logs, and combined analysis tables are available from the Dryad archive rather than duplicated in this repository.
- Software/scripts are MIT licensed; synthetic notes and annotation CSVs are CC BY 4.0 licensed.

## License

This repository uses a dual-license structure:

- Software and scripts are licensed under the MIT License; see `LICENSE`.
- Synthetic clinical notes and gold-standard annotation CSV files are licensed under CC BY 4.0; see `DATA_LICENSE.md`.
