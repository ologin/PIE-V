# Video Generation Examples

The video-generation material is organized as case studies for representative
PIE-V mistakes and corrections.

Public `cvprw26` examples:

- `georgiatech_covid_18_10`: wrong execution in a COVID-test procedure.
- `sfu_covid_002_10`: instruction-reading error followed by an explicit correction.

Published helper scripts:

- `scripts/video_generator_cmu_bike15_3_seedance.py`
- `scripts/video_generator_georgiatech_covid_18_10_seedance.py`
- `scripts/video_generator_sfu_cooking_008_5_kling.py`

Provider credentials are read from environment variables:

```bash
export ARK_API_KEY="..."
export KLING_ACCESS_KEY="..."
export KLING_SECRET_KEY="..."
```

The text planner covers all five PIE-V error types. The video scripts document
selected editing windows rather than a provider-agnostic generation API.
