# Sources and data dictionary

## Primary data source
- World Bank API indicator documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/898599-indicator-api-queries
- World Bank Data: https://data.worldbank.org/
- API endpoint pattern used by the app: `https://api.worldbank.org/v2/country/{ISO3}/indicator/{INDICATOR_CODE}?date={START}:{END}&format=json&per_page=20000`
- API data is retrieved at runtime. The app reports actual observed coverage and observation count. Missing values are dropped and are not replaced with invented values.

## Required indicators
| Display label | World Bank code | Units / definition |
|---|---|---|
| Life expectancy at birth | SP.DYN.LE00.IN | Years; total population |
| Unemployment rate (national estimate) | SL.UEM.TOTL.NE.ZS | % of total labor force; national estimate |
| GDP per capita | NY.GDP.PCAP.CD | Current US$ per person; GDP divided by population |
| Crude birth rate | SP.DYN.CBRT.IN | Births per 1,000 people |
| Net migration | SM.POP.NETM | People; immigration minus emigration for the source period |
| Intentional homicide rate | VC.IHR.PSRC.P5 | Intentional homicides per 100,000 people |
| Consumer price inflation | FP.CPI.TOTL.ZG | Annual percentage change |
| Central government debt | GC.DOD.TOTL.GD.ZS | % of GDP; central government only |
| Total reserves, including gold | FI.RES.TOTL.CD | Current US$ |
| External balance: goods and services | NE.RSB.GNFS.ZS | % of GDP; not current account |

## Country coverage
The app offers Mexico (MEX), Brazil (BRA), Argentina (ARG), Colombia (COL), Chile (CHL), Peru (PER), Venezuela (VEN), Ecuador (ECU), and Uruguay (URY). Brazil is a Latin American comparison, not a Spanish-speaking Hispanic country.

## Historical/context sources for student research
- World Bank country data and country reports: https://www.worldbank.org/en/where-we-work
- International Monetary Fund country pages and reports: https://www.imf.org/en/Countries
- United Nations Data: https://data.un.org/
- For Mexico GDP per capita, World Bank series page: https://data.worldbank.org/indicator/NY.GDP.PCAP.CD?locations=MX

## Model notes
The app uses a centered/scaled time variable `z = (year - base_year) / 10` and fits a polynomial of the selected degree with NumPy. R-squared and mean absolute error are in-sample metrics. A chronological holdout metric is reported separately when enough data are available. The app does not label residual margins as validated 95% prediction intervals.
