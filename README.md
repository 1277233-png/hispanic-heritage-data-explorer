# Hispanic Heritage Historical Data Explorer

A Streamlit app for investigating real historical indicators for Latin American countries using the World Bank API and polynomial regression.

## Featured investigation
Default setup: Mexico GDP per capita, with Colombia as a same-indicator comparison. GDP per capita is GDP divided by population, measured here in current US dollars per person. It is not average salary or inflation-adjusted income.

## Features
- Nine country choices: Mexico, Brazil, Argentina, Colombia, Chile, Peru, Venezuela, Ecuador, and Uruguay.
- Ten World Bank indicators with official codes, definitions, and units.
- Live official World Bank API retrieval for the last 70 completed calendar years (the actual data coverage can be shorter).
- Editable working data separate from original observations, validation, and Reset to source.
- Polynomial regression degree 3–8 using centered/scaled time, with equation, in-sample R-squared, MAE, mean, residuals, and coverage.
- Degree comparison and chronological holdout check when enough observations exist.
- Scatter plot, fitted curve, model sample markers, historical boundary, extrapolated segment, and same-indicator comparison countries.
- Function analysis: numerical turning points, possible inflection points, zeros within historical domain, and fastest modeled increase/decrease.
- Single-year and batch estimates (range or custom years).
- Genuine CSV, Excel workbook, and PDF report downloads.
- Printer-friendly summary mode and visible limitations.

## Run locally
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, `README.md`, and `sources.md` to a GitHub repository.
2. Open https://share.streamlit.io and connect the repository.
3. Select `app.py` as the main file and deploy.
4. Test data retrieval, changing country/indicator, editing/reset, regression, comparisons, estimates, and all downloads.

## Deploy to Render
- Create a Python Web Service connected to the repository.
- Build command: `pip install -r requirements.txt`
- Start command: `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

## Verification checklist
- Compare at least two source values with the linked official World Bank indicator page.
- Check actual coverage, units, country, indicator code, and retrieval date.
- Edit one value; verify the fit changes; reset and verify the source data return.
- Test duplicate years, nonnumeric values, blank rows, and too few distinct years.
- Independently check one model estimate and verify a batch estimate agrees with the single-year estimate for the same year.
- Test a year inside the observed range, a near-future year, and a distant future year.
- Change degree, marker interval, extrapolation horizon, comparison country, and printer-friendly mode.
- Open the CSV, XLSX, and PDF and confirm that each contains current selection/model details.
- Test an indicator/country with missing data and simulate an API failure by temporarily disconnecting network access; the app should explain the failure rather than invent data.

## Known limitations
- World Bank indicators may have missing years or end before the latest calendar year.
- National unemployment series may use different methods across countries; comparisons require care.
- Polynomial regression is descriptive and may extrapolate poorly. R-squared and MAE are in-sample fit metrics, not forecasting guarantees.
- No validated 95% prediction intervals are provided.
- Current US dollars are not inflation-adjusted and are affected by currency exchange rates.
- The app is a teaching tool; users must verify data and calculations and explain their own contributions.

## Team and AI-use record
Fill this section in before submission:
- Team members and roles:
- Featured question:
- Individual contribution:
- One independently checked calculation:
- One bug/debugging decision:
- AI assistance used and how outputs were verified:
- Live URL:
- Repository URL:
