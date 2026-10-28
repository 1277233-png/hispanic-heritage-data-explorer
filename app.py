import io
from datetime import date, datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

APP_TITLE = "Hispanic Heritage Historical Data Explorer"
API_ROOT = "https://api.worldbank.org/v2"

COUNTRIES = {
    "Mexico": "MEX", "Brazil": "BRA", "Argentina": "ARG", "Colombia": "COL",
    "Chile": "CHL", "Peru": "PER", "Venezuela": "VEN", "Ecuador": "ECU",
    "Uruguay": "URY"
}
INDICATORS = {
    "Life expectancy at birth": {
        "code": "SP.DYN.LE00.IN", "units": "years",
        "definition": "Life expectancy at birth, total population."
    },
    "Unemployment rate (national estimate)": {
        "code": "SL.UEM.TOTL.NE.ZS", "units": "% of total labor force",
        "definition": "Unemployment as a percentage of the total labor force, national estimate."
    },
    "GDP per capita": {
        "code": "NY.GDP.PCAP.CD", "units": "current US$ per person",
        "definition": "Gross domestic product divided by population, in current US dollars; not average salary or inflation-adjusted income."
    },
    "Crude birth rate": {
        "code": "SP.DYN.CBRT.IN", "units": "births per 1,000 people",
        "definition": "Annual number of births per 1,000 people, not a count of births."
    },
    "Net migration": {
        "code": "SM.POP.NETM", "units": "people",
        "definition": "Immigration minus emigration over the source period; not the number of people leaving."
    },
    "Intentional homicide rate": {
        "code": "VC.IHR.PSRC.P5", "units": "homicides per 100,000 people",
        "definition": "Intentional homicide rate per 100,000 people."
    },
    "Consumer price inflation": {
        "code": "FP.CPI.TOTL.ZG", "units": "annual %",
        "definition": "Annual percentage change in the consumer price index."
    },
    "Central government debt": {
        "code": "GC.DOD.TOTL.GD.ZS", "units": "% of GDP",
        "definition": "Central government debt as a percentage of GDP; not all public-sector debt."
    },
    "Total reserves, including gold": {
        "code": "FI.RES.TOTL.CD", "units": "current US$",
        "definition": "Total reserves including gold, in current US dollars."
    },
    "External balance: goods and services": {
        "code": "NE.RSB.GNFS.ZS", "units": "% of GDP",
        "definition": "External balance on goods and services as a percentage of GDP; not the current account."
    },
}

START_YEAR = date.today().year - 70
END_YEAR = date.today().year - 1

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
st.title("📈 " + APP_TITLE)
st.caption("A student-friendly tool for exploring real historical indicators from Latin America using polynomial regression.")
st.info(
    "Brazil is included as a Latin American comparison; Portuguese is Brazil's official language, so Brazil is not a Spanish-speaking Hispanic country. "
    "A polynomial is a descriptive model, not proof of causes or a guaranteed forecast."
)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_indicator(country_code: str, indicator_code: str, start_year: int, end_year: int):
    url = f"{API_ROOT}/country/{country_code}/indicator/{indicator_code}"
    params = {"date": f"{start_year}:{end_year}", "format": "json", "per_page": 20000}
    try:
        response = requests.get(url, params=params, timeout=25)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"World Bank request failed: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("The World Bank returned a response that was not valid JSON.") from exc
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError("Unexpected World Bank API response format.")
    rows = payload[1]
    if rows is None:
        return pd.DataFrame(columns=["Year", "Value"])
    cleaned = []
    for row in rows:
        if row and row.get("date") and row.get("value") is not None:
            try:
                yr = int(row["date"])
                val = float(row["value"])
                if start_year <= yr <= end_year and np.isfinite(val):
                    cleaned.append((yr, val))
            except (ValueError, TypeError):
                continue
    df = pd.DataFrame(cleaned, columns=["Year", "Value"])
    if df.empty:
        return df
    return df.drop_duplicates("Year").sort_values("Year").reset_index(drop=True)

def fit_polynomial(df: pd.DataFrame, degree: int):
    data = df.dropna(subset=["Year", "Value"]).copy()
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
    data = data.dropna().sort_values("Year")
    data = data.drop_duplicates("Year")
    if len(data) < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct years; only {len(data)} are available.")
    base_year = int(round(float(data["Year"].mean())))
    # Scale centered time by decade to keep coefficients readable and numerical fitting stable.
    z = (data["Year"].to_numpy(dtype=float) - base_year) / 10.0
    y = data["Value"].to_numpy(dtype=float)
    coeffs = np.polyfit(z, y, degree)
    model = np.poly1d(coeffs)
    fitted = model(z)
    residuals = y - fitted
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - np.sum(residuals ** 2) / sst) if sst > 0 else float("nan")
    mae = float(np.mean(np.abs(residuals)))
    data["Fitted"] = fitted
    data["Residual"] = residuals
    return {"model": model, "coeffs": coeffs, "base_year": base_year, "data": data,
            "r2": r2, "mae": mae, "residuals": residuals,
            "min_year": int(data["Year"].min()), "max_year": int(data["Year"].max()),
            "mean": float(data["Value"].mean()), "degree": degree}

def model_value(fit, year):
    z = (float(year) - fit["base_year"]) / 10.0
    return float(fit["model"](z))

def equation_string(fit):
    terms = []
    degree = fit["degree"]
    for i, coefficient in enumerate(fit["coeffs"]):
        power = degree - i
        if abs(coefficient) < 1e-12:
            continue
        if power == 0:
            term = f"{coefficient:.8g}"
        elif power == 1:
            term = f"{coefficient:.8g}z"
        else:
            term = f"{coefficient:.8g}z^{power}"
        terms.append(term)
    return "y = " + " + ".join(terms).replace("+ -", "- ") + f", where z = (year - {fit['base_year']}) / 10"

def make_chart(df, fit, title, units, sample_step=1, compare_df=None, compare_name=None, horizon=5):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(df["Year"], df["Value"], s=22, alpha=0.85, label="Observed data")
    historical_years = np.linspace(fit["min_year"], fit["max_year"], 350)
    historical_values = [model_value(fit, yr) for yr in historical_years]
    ax.plot(historical_years, historical_values, linewidth=2.3, label=f"Degree {fit['degree']} fit")
    sampled_years = np.arange(fit["min_year"], fit["max_year"] + 1, sample_step)
    ax.scatter(sampled_years, [model_value(fit, yr) for yr in sampled_years], s=18, marker="x", label=f"Model markers every {sample_step} year(s)")
    if horizon > 0:
        extra_years = np.linspace(fit["max_year"], fit["max_year"] + horizon, max(20, horizon * 8))
        ax.plot(extra_years, [model_value(fit, yr) for yr in extra_years], linestyle="--", linewidth=2, label="Extrapolation")
        ax.axvline(fit["max_year"], linestyle=":", linewidth=1.3, label=f"Last observed year ({fit['max_year']})")
    if compare_df is not None and not compare_df.empty and compare_name:
        ax.plot(compare_df["Year"], compare_df["Value"], marker=".", linewidth=1.5, label=f"{compare_name} observed")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(units)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

def create_xlsx(meta, data, results, summary):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        data.to_excel(writer, sheet_name="Working Data", index=False)
        if results is not None and not results.empty:
            results.to_excel(writer, sheet_name="Calculated Years", index=False)
        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#234B6B", "font_color": "white", "text_wrap": True})
        for sheet_name, frame in [("Summary", summary), ("Working Data", data)] + ([("Calculated Years", results)] if results is not None and not results.empty else []):
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(1, len(frame)), max(0, len(frame.columns)-1))
            for col_idx, col in enumerate(frame.columns):
                ws.write(0, col_idx, col, header_fmt)
                width = min(max(12, len(str(col)) + 2, int(frame[col].astype(str).str.len().quantile(0.9)) + 2 if len(frame) else 12), 42)
                ws.set_column(col_idx, col_idx, width)
    output.seek(0)
    return output.getvalue()

def create_pdf(title, meta_lines, equation, metrics, findings, chart_bytes, data_preview, results):
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=landscape(letter), rightMargin=0.45*inch, leftMargin=0.45*inch, topMargin=0.4*inch, bottomMargin=0.4*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    for line in meta_lines:
        story.append(Paragraph(line, styles["BodyText"]))
    story.extend([Spacer(1, 6), Paragraph("Regression equation", styles["Heading2"]), Paragraph(equation, styles["BodyText"]), Spacer(1, 5)])
    for line in metrics:
        story.append(Paragraph(line, styles["BodyText"]))
    story.extend([Spacer(1, 6), Paragraph("Graph", styles["Heading2"])])
    if chart_bytes:
        img_buf = io.BytesIO(chart_bytes)
        story.append(Image(img_buf, width=7.4*inch, height=4.0*inch))
    story.append(Paragraph("Findings and limitations", styles["Heading2"]))
    for item in findings:
        story.append(Paragraph("• " + item, styles["BodyText"]))
    story.extend([Spacer(1, 6), Paragraph("Data preview", styles["Heading2"])])
    preview = [list(data_preview.columns)] + data_preview.head(10).astype(str).values.tolist()
    table = Table(preview, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#234B6B")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(table)
    if results is not None and not results.empty:
        story.extend([Spacer(1, 8), Paragraph("Calculated years", styles["Heading2"])])
        res_preview = [list(results.columns)] + results.head(20).astype(str).values.tolist()
        t = Table(res_preview, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#234B6B")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 7),
        ]))
        story.append(t)
    doc.build(story)
    output.seek(0)
    return output.getvalue()

with st.sidebar:
    st.header("Model controls")
    country = st.selectbox("Main country", list(COUNTRIES.keys()), index=0)
    indicator_label = st.selectbox("Indicator", list(INDICATORS.keys()), index=2)
    degree = st.slider("Polynomial degree", min_value=3, max_value=8, value=3)
    sample_step = st.slider("Model marker interval (years)", min_value=1, max_value=10, value=5)
    horizon = st.slider("Forward display horizon (years)", min_value=0, max_value=50, value=5)
    comparisons = st.multiselect("Comparison countries (same indicator)", [c for c in COUNTRIES if c != country], default=["Colombia"] if country != "Colombia" else ["Mexico"])
    printer_mode = st.toggle("Printer-friendly summary", value=False)
    if st.button("Refresh live data"):
        st.cache_data.clear()
        st.rerun()

indicator = INDICATORS[indicator_label]
start_year = date.today().year - 70
end_year = date.today().year - 1
st.caption(f"Requested window: {start_year}–{end_year}. Actual coverage depends on official data availability.")
try:
    with st.spinner(f"Retrieving {indicator_label} data for {country} from the World Bank…"):
        source_df = fetch_indicator(COUNTRIES[country], indicator["code"], start_year, end_year)
except RuntimeError as exc:
    st.error(str(exc))
    st.markdown(f"[Open the official World Bank series](https://data.worldbank.org/indicator/{indicator['code']}?locations={COUNTRIES[country]})")
    st.stop()

if source_df.empty:
    st.warning("The World Bank API returned no observations for this country, indicator, and period. No replacement or synthetic data will be created.")
    st.markdown(f"[Check the official series](https://data.worldbank.org/indicator/{indicator['code']}?locations={COUNTRIES[country]})")
    st.stop()

if "original_data" not in st.session_state or st.session_state.get("source_key") != (country, indicator["code"], start_year, end_year):
    st.session_state.original_data = source_df.copy()
    st.session_state.working_data = source_df.copy()
    st.session_state.source_key = (country, indicator["code"], start_year, end_year)
    st.session_state.batch_results = pd.DataFrame()

st.subheader(f"{country}: {indicator_label}")
st.write(f"**Definition:** {indicator['definition']}")
st.write(f"**Units:** {indicator['units']}  |  **World Bank indicator code:** `{indicator['code']}`")
st.write(f"**Source:** [World Bank indicator page](https://data.worldbank.org/indicator/{indicator['code']}?locations={COUNTRIES[country]})  |  **Retrieved:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.write(f"**Actual coverage:** {int(source_df['Year'].min())}–{int(source_df['Year'].max())} | **Observations:** {len(source_df)}")

st.markdown("### Editable working data")
st.caption("Edit years or values to test a scenario. The original source data remain available through Reset to source.")
edited = st.data_editor(
    st.session_state.working_data,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Year": st.column_config.NumberColumn("Year", min_value=1900, max_value=2100, step=1, format="%d"),
        "Value": st.column_config.NumberColumn(f"Value ({indicator['units']})", format="%.6f"),
    },
    key=f"editor_{country}_{indicator['code']}"
)
col_reset, col_status = st.columns([1, 3])
with col_reset:
    if st.button("Reset to source"):
        st.session_state.working_data = st.session_state.original_data.copy()
        st.session_state.batch_results = pd.DataFrame()
        st.rerun()
with col_status:
    changed = not edited.reset_index(drop=True).equals(st.session_state.original_data.reset_index(drop=True))
    st.write("**Data status:** User-modified scenario" if changed else "**Data status:** Original source data")

# Validate edited working data before fitting.
validation_errors = []
work = edited.copy()
if not {"Year", "Value"}.issubset(work.columns):
    validation_errors.append("The table must contain Year and Value columns.")
else:
    work["Year"] = pd.to_numeric(work["Year"], errors="coerce")
    work["Value"] = pd.to_numeric(work["Value"], errors="coerce")
    if work[["Year", "Value"]].isna().any().any():
        validation_errors.append("Every retained row needs a numeric year and value. Remove blank rows or fill them.")
    if not validation_errors:
        if not np.all(np.equal(work["Year"], np.floor(work["Year"]))):
            validation_errors.append("Years must be integers.")
        if work["Year"].duplicated().any():
            validation_errors.append("Duplicate years are not allowed.")
        if not np.isfinite(work["Value"]).all():
            validation_errors.append("Values must be finite numbers.")
        if indicator_label in ["Life expectancy at birth", "Crude birth rate", "Intentional homicide rate"] and (work["Value"] < 0).any():
            validation_errors.append("Negative values are not valid for this indicator.")
        if indicator_label == "Life expectancy at birth" and (work["Value"] > 125).any():
            validation_errors.append("Life expectancy above 125 years is likely invalid; verify the value.")
        if indicator_label == "Unemployment rate (national estimate)" and ((work["Value"] < 0) | (work["Value"] > 100)).any():
            validation_errors.append("Unemployment percentages must be between 0 and 100.")
        if indicator_label == "Central government debt" and (work["Value"] < 0).any():
            validation_errors.append("Negative central government debt percentage is not accepted by this app's basic validation.")
if validation_errors:
    for err in validation_errors:
        st.error(err)
    st.stop()

work = work.sort_values("Year").reset_index(drop=True)
if len(work) < degree + 1:
    st.error(f"Degree {degree} requires at least {degree+1} distinct years. Current valid rows: {len(work)}.")
    st.stop()

try:
    fit = fit_polynomial(work, degree)
except (ValueError, np.linalg.LinAlgError) as exc:
    st.error(f"Could not fit this model: {exc}")
    st.stop()

st.markdown("### Regression model")
eq = equation_string(fit)
st.code(eq)
m1, m2, m3, m4 = st.columns(4)
m1.metric("R² (in-sample)", f"{fit['r2']:.4f}")
m2.metric("Mean absolute error", f"{fit['mae']:,.3f} {indicator['units']}")
m3.metric("Observations", f"{len(fit['data'])}")
m4.metric("Fitted range", f"{fit['min_year']}–{fit['max_year']}")
st.caption("R² and MAE describe the fit to the data used to estimate the model; they do not prove forecast accuracy.")

# Fit a comparison degree for the required degree comparison.
comparison_degree = 5 if degree != 5 else 3
degree_compare = fit_polynomial(work, comparison_degree) if len(work) >= comparison_degree + 1 else None
if degree_compare:
    st.write(f"**Degree comparison:** degree {degree} R² = {fit['r2']:.4f}, MAE = {fit['mae']:.3f}; degree {comparison_degree} R² = {degree_compare['r2']:.4f}, MAE = {degree_compare['mae']:.3f}. Higher degree can improve in-sample fit while worsening extrapolation.")

# Chronological holdout: reserve latest up to 5 observations, only if enough training rows remain.
holdout_n = min(5, max(0, len(work) - (degree + 2)))
if holdout_n >= 3:
    train_df = work.iloc[:-holdout_n].copy()
    test_df = work.iloc[-holdout_n:].copy()
    if len(train_df) >= degree + 1:
        try:
            hold_fit = fit_polynomial(train_df, degree)
            hold_predictions = np.array([model_value(hold_fit, yr) for yr in test_df["Year"]])
            hold_mae = float(np.mean(np.abs(test_df["Value"].to_numpy() - hold_predictions)))
            st.write(f"**Chronological holdout MAE:** {hold_mae:.3f} {indicator['units']} using the latest {holdout_n} observations as test data. This score is separate from the in-sample metrics.")
            with st.expander("View holdout results"):
                hold_display = test_df[["Year", "Value"]].copy()
                hold_display["Model estimate"] = hold_predictions
                hold_display["Absolute error"] = np.abs(hold_display["Value"] - hold_display["Model estimate"])
                st.dataframe(hold_display, use_container_width=True)
        except (ValueError, np.linalg.LinAlgError):
            st.caption("Holdout test unavailable because the training data are insufficient for this degree.")
else:
    st.caption("Chronological holdout test unavailable: not enough observations remain for both training and testing at this degree.")

# Comparison data
comparison_frames = {}
for comp_country in comparisons:
    try:
        comp_df = fetch_indicator(COUNTRIES[comp_country], indicator["code"], start_year, end_year)
        if not comp_df.empty:
            comparison_frames[comp_country] = comp_df
    except RuntimeError as exc:
        st.warning(f"Comparison data for {comp_country} unavailable: {exc}")

st.markdown("### Graph: observed points, fitted curve, and extrapolation")
if printer_mode:
    st.subheader("Printer-friendly report summary")
    st.write(f"{country} | {indicator_label} | {indicator['units']} | Degree {degree}")
    st.write(f"Equation: {eq}")
    st.write(f"R² = {fit['r2']:.4f}; MAE = {fit['mae']:.3f}; coverage = {fit['min_year']}–{fit['max_year']}.")
fig = make_chart(work, fit, f"{country}: {indicator_label}", indicator["units"], sample_step, horizon=horizon)
for comp_country, comp_df in comparison_frames.items():
    fig.axes[0].plot(comp_df["Year"], comp_df["Value"], marker=".", linewidth=1.4, label=f"{comp_country} observed")
fig.axes[0].legend(fontsize=8)
st.pyplot(fig, use_container_width=True)
chart_bytes_io = io.BytesIO()
fig.savefig(chart_bytes_io, format="png", dpi=160, bbox_inches="tight")
chart_bytes = chart_bytes_io.getvalue()
plt.close(fig)
for comp_country, comp_df in comparison_frames.items():
    st.caption(f"{comp_country} coverage: {int(comp_df['Year'].min())}–{int(comp_df['Year'].max())} ({len(comp_df)} observations). Only same-indicator official values are plotted; missing years are not filled.")

st.markdown("### Function analysis")
st.write(f"- **Domain used for historical interpretation:** {fit['min_year']} to {fit['max_year']}. The polynomial is mathematically defined for all real inputs, but values outside this interval are extrapolations.")
st.write(f"- **Model mean:** {fit['mean']:.4f} {indicator['units']}.")
st.write("- **Turning points and increasing/decreasing intervals:** use the graph and the derivative-based analysis below where available. These are properties of the fitted polynomial, not direct observations.")
# Derivative-based numerical analysis in the scaled z coordinate.
deriv = np.polyder(fit["model"])
second = np.polyder(deriv)
zlo = (fit["min_year"] - fit["base_year"]) / 10.0
zhi = (fit["max_year"] - fit["base_year"]) / 10.0
crit_z = [float(r.real) for r in np.roots(deriv) if abs(r.imag) < 1e-8 and zlo <= r.real <= zhi]
infl_z = [float(r.real) for r in np.roots(second) if abs(r.imag) < 1e-8 and zlo <= r.real <= zhi]
if crit_z:
    crit_years = [fit["base_year"] + 10*t for t in crit_z]
    st.write("**Critical points inside the historical domain:**")
    for t, yr in zip(crit_z, crit_years):
        val = float(fit["model"](t))
        curv = float(second(t))
        kind = "local minimum" if curv > 0 else ("local maximum" if curv < 0 else "unclassified/flat")
        st.write(f"- {kind.title()} near year {yr:.2f}, modeled value {val:.4g} {indicator['units']}.")
else:
    st.write("- No interior critical points were found in the historical interval.")
if infl_z:
    st.write("**Possible inflection points** (second derivative zero inside the interval): " + ", ".join(f"{fit['base_year'] + 10*t:.2f}" for t in infl_z))
else:
    st.write("**Possible inflection points:** none found inside the historical interval.")
# Derivative in units per year; z changes 0.1 per year.
rate_candidates = [zlo, zhi] + [float(r.real) for r in np.roots(second) if abs(r.imag) < 1e-8 and zlo <= r.real <= zhi]
rates = [(fit["base_year"] + 10*t, float(deriv(t)/10.0)) for t in rate_candidates]
if rates:
    fastest_up = max(rates, key=lambda item: item[1])
    fastest_down = min(rates, key=lambda item: item[1])
    st.write(f"- **Fastest modeled increase** on the historical interval: around {fastest_up[0]:.2f}, at {fastest_up[1]:.4g} {indicator['units']} per year.")
    st.write(f"- **Fastest modeled decrease** on the historical interval: around {fastest_down[0]:.2f}, at {fastest_down[1]:.4g} {indicator['units']} per year.")
roots = np.roots(fit["model"])
real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-8 and zlo <= r.real <= zhi]
if real_roots:
    st.write("**Zeros within historical interval:** " + ", ".join(f"{fit['base_year'] + 10*r:.2f}" for r in real_roots) + ". Interpret as modeled output equal to zero in the selected units.")
else:
    st.write("**Zeros within historical interval:** none found.")
st.caption("The numerical turning-point calculations are for the fitted polynomial. Sparse annual data do not prove the real-world process changes smoothly between observations.")

st.markdown("### Single-year estimate")
single_year = st.number_input("Year to estimate", min_value=1950, max_value=2100, value=min(END_YEAR + 1, 2100), step=1)
if st.button("Calculate value"):
    est = model_value(fit, int(single_year))
    if fit["min_year"] <= single_year <= fit["max_year"]:
        calc_type = "Fitted value within observed data range"
    else:
        calc_type = "Extrapolation outside observed data range"
    actual_row = work.loc[work["Year"] == single_year, "Value"]
    st.session_state.single_result = {
        "Year": int(single_year), "Model estimate": est, "Units": indicator["units"], "Calculation type": calc_type,
        "Observed value": float(actual_row.iloc[0]) if not actual_row.empty else np.nan
    }
if "single_result" in st.session_state:
    sr = st.session_state.single_result
    st.write(f"**{sr['Year']}:** {sr['Model estimate']:,.6g} {sr['Units']} — {sr['Calculation type']}.")
    if pd.notna(sr["Observed value"]):
        st.write(f"Observed value for that year: {sr['Observed value']:,.6g} {sr['Units']}.")
    else:
        st.write("No observed value exists in the working table for this year.")
    if sr["Year"] > fit["max_year"] + 10 or sr["Year"] < fit["min_year"] - 10:
        st.warning("This estimate is far outside the fitted range. Polynomial extrapolation can become implausible quickly.")

st.markdown("### Batch estimates")
batch_mode = st.radio("Batch input type", ["Year range", "Custom years"], horizontal=True)
years_to_calc = []
if batch_mode == "Year range":
    c1, c2, c3 = st.columns(3)
    with c1:
        batch_start = st.number_input("Start year", min_value=1950, max_value=2100, value=max(1950, fit["min_year"]), step=1)
    with c2:
        batch_end = st.number_input("End year", min_value=1950, max_value=2100, value=min(2100, fit["max_year"] + 5), step=1)
    with c3:
        batch_step = st.number_input("Step (1–10 years)", min_value=1, max_value=10, value=1, step=1)
    if batch_end <= batch_start:
        st.error("End year must be greater than start year.")
    else:
        years_to_calc = list(range(int(batch_start), int(batch_end) + 1, int(batch_step)))
else:
    custom_text = st.text_input("Enter years separated by commas", value=f"{fit['max_year']}, {fit['max_year']+1}, {fit['max_year']+5}")
    raw_parts = [p.strip() for p in custom_text.split(",") if p.strip()]
    parsed = []
    invalid = []
    for part in raw_parts:
        try:
            yr = int(part)
            if 1950 <= yr <= 2100:
                parsed.append(yr)
            else:
                invalid.append(part)
        except ValueError:
            invalid.append(part)
    if invalid:
        st.warning("Ignored invalid/out-of-range entries: " + ", ".join(invalid))
    years_to_calc = list(dict.fromkeys(parsed))
st.write("**Valid years to calculate:** " + (", ".join(map(str, years_to_calc)) if years_to_calc else "None"))
if st.button("Calculate batch", disabled=not years_to_calc):
    rows = []
    for yr in years_to_calc:
        val = model_value(fit, yr)
        observed = work.loc[work["Year"] == yr, "Value"]
        if yr < fit["min_year"] or yr > fit["max_year"]:
            label = "Extrapolation"
        elif not observed.empty:
            label = "Fitted value at observed year"
        else:
            label = "Within-range estimate"
        rows.append({"Year": yr, "Estimate": val, "Units": indicator["units"], "Type": label,
                     "Observed value": float(observed.iloc[0]) if not observed.empty else np.nan})
    st.session_state.batch_results = pd.DataFrame(rows)
if isinstance(st.session_state.get("batch_results"), pd.DataFrame) and not st.session_state.batch_results.empty:
    st.dataframe(st.session_state.batch_results, use_container_width=True)

# Exports
st.markdown("### Download reports")
meta = {
    "Country": country, "Indicator": indicator_label, "Indicator code": indicator["code"],
    "Units": indicator["units"], "Source": "World Bank API", "Retrieval timestamp": datetime.now().isoformat(timespec="minutes"),
    "Actual coverage": f"{fit['min_year']}–{fit['max_year']}", "Observations": len(work),
    "Degree": degree, "Base year": fit["base_year"], "Equation": eq, "R-squared (in-sample)": fit["r2"],
    "MAE": fit["mae"], "Data status": "User-modified scenario" if changed else "Original source data",
    "Mean": fit["mean"]
}
working_export = work[["Year", "Value"]].copy()
working_export.columns = ["Year", f"Value ({indicator['units']})"]
working_export["Data status"] = meta["Data status"]
summary_df = pd.DataFrame({"Item": list(meta.keys()), "Value": list(meta.values())})
results_df = st.session_state.batch_results if isinstance(st.session_state.get("batch_results"), pd.DataFrame) else pd.DataFrame()
csv_bytes = working_export.to_csv(index=False).encode("utf-8")
xlsx_bytes = create_xlsx(meta, working_export, results_df, summary_df)
pdf_meta = [
    f"Country: {country} | Indicator: {indicator_label} ({indicator['code']})",
    f"Units: {indicator['units']} | Source: World Bank API | Retrieved: {meta['Retrieval timestamp']}",
    f"Coverage: {fit['min_year']}–{fit['max_year']} | Observations: {len(work)} | Degree: {degree} | Base year: {fit['base_year']}",
    f"Data status: {meta['Data status']}"
]
pdf_metrics = [f"R-squared (in-sample): {fit['r2']:.5f}", f"MAE: {fit['mae']:.5g} {indicator['units']}", f"Mean: {fit['mean']:.5g} {indicator['units']}"]
pdf_findings = [
    "The regression describes the selected historical data and is not proof of causation.",
    "Polynomial extrapolation may become implausible outside the observed range.",
    "GDP per capita in current US dollars is not inflation-adjusted and can move with exchange rates.",
    "No statistically validated 95% prediction intervals are provided."
]
pdf_bytes = create_pdf(f"{country}: {indicator_label} — Function Analysis Report", pdf_meta, eq, pdf_metrics, pdf_findings, chart_bytes, working_export, results_df)
d1, d2, d3 = st.columns(3)
with d1:
    st.download_button("Download CSV", csv_bytes, file_name=f"{country.lower()}_{indicator['code'].replace('.', '_')}_data.csv", mime="text/csv")
with d2:
    st.download_button("Download Excel workbook", xlsx_bytes, file_name=f"{country.lower()}_{indicator['code'].replace('.', '_')}_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with d3:
    st.download_button("Download PDF report", pdf_bytes, file_name=f"{country.lower()}_{indicator['code'].replace('.', '_')}_report.pdf", mime="application/pdf")

st.markdown("### Historical context and interpretation")
st.write("Use credible country-specific historical sources to investigate changes visible in the data. Correlation and timing alone do not establish that a historical event caused a change in the indicator.")
st.markdown("""
- [World Bank Data API documentation](https://datahelpdesk.worldbank.org/knowledgebase/articles/898599-indicator-api-queries)
- [World Bank DataBank](https://data.worldbank.org/)
- [World Bank country profiles](https://www.worldbank.org/en/where-we-work)
- [International Monetary Fund country information](https://www.imf.org/en/Countries)
- [United Nations Data](https://data.un.org/)
""")
with st.expander("Precalculus analysis checklist"):
    st.markdown("""
    - Compare at least two polynomial degrees using equations, graphs, in-sample R²/MAE, and residual patterns.
    - Compute average rates of change from clearly identified observed or modeled values.
    - Explain turning points, increasing/decreasing intervals, and concavity using the fitted curve and changing average rates.
    - Include one within-range estimate, one near-future extrapolation, and one distant extrapolation.
    - Report three supported findings, two limitations, and one follow-up question in your written report.
    - Keep a record of source checks, edits/reset, single/batch agreement, comparison behavior, and export tests.
    """)
st.caption("Source integrity note: missing official data are reported as missing. This app does not fabricate replacement observations.")
